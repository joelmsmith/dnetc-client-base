/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 cores for Tenstorrent devices. All five RISC-V cores of every
 * Tensix tile run persistent firmware (r72tt_tile.c); with the SFPU cores,
 * TRISC1 drives the tile's Vector Unit instead of checking keys itself,
 * and with the RVV core TRISC2 also checks keys with its own vector unit
 * and the Vector Unit is driven with SFPLOADMACRO.
 * Each call splits the keys into one contiguous range per core ("worker"),
 * sized by the worker's speed, and merges the per-worker results in key
 * order, so the outcome matches a sequential core exactly.
 *
 * Setting DNETC_TT_STATS=1 logs device clock statistics for every call.
*/

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "cputypes.h"
#include "ccoreio.h"
#include "logstuff.h"
#include "triggers.h"
#include "tt_setup.h"
#include "r72tt_fw.h"

extern "C" s32 rc5_72_unit_func_tt_riscv (RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_tt_sfpu (RC5_72UnitWork *, u32 *, void *);
extern "C" s32 rc5_72_unit_func_tt_sfpu_rvv (RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL rc5_72_unit_func_ansi_1 (RC5_72UnitWork *, u32 *, void *);

/* Firmware image (r72tt_tile_fw.cpp), generated from the firmware sources
   by 'make tt-firmware' */
extern "C" const unsigned char r72tt_tile_fw[];
extern "C" const unsigned int r72tt_tile_fw_size;

/* Generous bound on device clocks per key, to detect a hung device */
#define R72TT_MAX_CLOCKS_PER_KEY  50000u
#define R72TT_POLL_CLOCKS         20000u
#define R72TT_BOOT_CLOCKS         1000000u

#define R72TT_MAX_WORKERS (TT_MAX_TILES * R72TT_CORES)

/* Worker classes, and their initial speed estimates in clocks per key */
#define CLASS_SCALAR  0
#define CLASS_SFPU    1   /* TRISC1 driving the Vector Unit */
#define CLASS_RVV     2   /* TRISC2 using its RISC-V vector unit */
#define CLASS_SFPU_LM 3   /* CLASS_SFPU with SFPLOADMACRO */
#define CLASSES       4
static const double initial_clocks_per_key[CLASSES] = { 560.0, 35.0, 300.0, 32.0 };
static const u32 class_flags[CLASSES] = { 0, R72TT_FLAG_SFPU, R72TT_FLAG_RVV,
                                          R72TT_FLAG_SFPU | R72TT_FLAG_LOADMACRO };
static const char * const class_name[CLASSES] = { "RISC-V", "SFPU", "RVV", "SFPU (SFPLOADMACRO)" };

/* What the cores use besides the scalar code. MODE_SFPU_RVV is for
   hardware only, so its SFPU worker also uses SFPLOADMACRO. */
#define MODE_RISCV     0
#define MODE_SFPU      1
#define MODE_SFPU_RVV  2

static int mode_uses_class(int mode, int cls)
{
  return cls == CLASS_SCALAR || (cls == CLASS_SFPU && mode == MODE_SFPU) ||
         ((cls == CLASS_RVV || cls == CLASS_SFPU_LM) && mode == MODE_SFPU_RVV);
}

/* The class of RISC-V core 'core' of a tile */
static int worker_class(int mode, unsigned core)
{
  if (core == R72TT_CORE_TRISC1 && mode_uses_class(mode, CLASS_SFPU))
    return CLASS_SFPU;
  if (core == R72TT_CORE_TRISC1 && mode_uses_class(mode, CLASS_SFPU_LM))
    return CLASS_SFPU_LM;
  if (core == R72TT_CORE_TRISC2 && mode_uses_class(mode, CLASS_RVV))
    return CLASS_RVV;
  return CLASS_SCALAR;
}

typedef struct
{
  u32 worker;
  u32 offset, count;   /* range relative to the first key of the call */
  u32 hi, mid, lo;     /* first key */
} job_t;

/* Per device, and only used with the device locked: each device has its own
   cruncher thread */
typedef struct
{
  int booted;
  int checked[CLASSES];   /* self_check() passed, per worker class */
  u32 seq;
  double clocks_per_key[CLASSES];
  job_t jobs[R72TT_MAX_WORKERS];
  u32 workers[R72TT_MAX_WORKERS];
} device_state_t;

static device_state_t device_state[TT_MAX_DEVICES];
static int show_stats = -1;

#define MB_FIELD(core, f) (R72TT_MAILBOX(core) + (u32)offsetof(r72tt_mailbox_t, f))

/* Adds n to a key in the client's format (see the key increment in
   r72ansi1.cpp): hi is the least significant byte of the key counter,
   followed by mid and lo with their bytes in reverse order. */
static u32 swap32(u32 x)
{
  return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}

static void key_add(u32 *hi, u32 *mid, u32 *lo, u32 n)
{
  u32 h = *hi + (n & 0xFF);
  u32 m = swap32(*mid);
  u32 l = swap32(*lo);
  u32 add = (n >> 8) + (h >> 8);
  u32 m2 = m + add;

  *hi = h & 0xFF;
  if (m2 < m)
    l++;
  *mid = swap32(m2);
  *lo = swap32(l);
}

/* RC5_72UnitWork is packed, so its members are not passed by address */
static void work_key_add(RC5_72UnitWork *w, u32 n)
{
  u32 hi = w->L0.hi, mid = w->L0.mid, lo = w->L0.lo;
  key_add(&hi, &mid, &lo, n);
  w->L0.hi = hi;
  w->L0.mid = mid;
  w->L0.lo = lo;
}

static u32 rotl(u32 x, u32 s)
{
  s &= 31;
  return s ? (x << s) | (x >> (32 - s)) : x;
}

/* Straightforward RC5-72 encryption, for building self-check vectors */
static void rc5_72_encrypt(u32 hi, u32 mid, u32 lo, u32 plain_lo, u32 plain_hi,
                           u32 *cypher_lo, u32 *cypher_hi)
{
  u32 S[26], L[3], A = 0, B = 0;
  int i, k;

  L[0] = lo;
  L[1] = mid;
  L[2] = hi;
  for (S[0] = 0xB7E15163u, i = 1; i < 26; i++)
    S[i] = S[i - 1] + 0x9E3779B9u;
  for (k = 0; k < 3 * 26; k++)
  {
    A = S[k % 26] = rotl(S[k % 26] + A + B, 3);
    B = L[k % 3] = rotl(L[k % 3] + A + B, A + B);
  }
  A = plain_lo + S[0];
  B = plain_hi + S[1];
  for (i = 1; i <= 12; i++)
  {
    A = rotl(A ^ B, B) + S[2 * i];
    B = rotl(B ^ A, A) + S[2 * i + 1];
  }
  *cypher_lo = A;
  *cypher_hi = B;
}

/* Workers are numbered tile-major: worker w is core (w % R72TT_CORES) of
   tile (w / R72TT_CORES). */
static u32 worker_read32(tt_device_t *dev, unsigned w, u32 field_offset)
{
  return tt_l1_read32(dev, w / R72TT_CORES, R72TT_MAILBOX(w % R72TT_CORES) + field_offset);
}

/* Runs device clocks until the listed workers have a mailbox word equal to
   'value', or until 'budget' clocks have passed. Returns the clocks run, or
   -1 on timeout. */
static si64 wait_workers(tt_device_t *dev, const u32 *workers, unsigned n,
                         u32 field_offset, u32 value, ui64 budget)
{
  ui64 spent = 0;
  unsigned i = 0;

  for (;;)
  {
    while (i < n && worker_read32(dev, workers[i], field_offset) == value)
      i++;
    if (i == n)
      return (si64)spent;
    if (spent >= budget)
      return -1;
    tt_advance(dev, R72TT_POLL_CLOCKS);
    spent += R72TT_POLL_CLOCKS;
  }
}

/* Writes a job to its worker's mailbox and starts it */
static void post_job(tt_device_t *dev, const job_t *job, const RC5_72UnitWork *w,
                     u32 flags, u32 seq)
{
  const unsigned tile = job->worker / R72TT_CORES, core = job->worker % R72TT_CORES;
  r72tt_mailbox_t mb;

  mb.key_hi = job->hi;
  mb.key_mid = job->mid;
  mb.key_lo = job->lo;
  mb.count = job->count;
  mb.plain_lo = w->plain.lo;
  mb.plain_hi = w->plain.hi;
  mb.cypher_lo = w->cypher.lo;
  mb.cypher_hi = w->cypher.hi;
  mb.flags = flags;
  tt_l1_write(dev, tile, MB_FIELD(core, key_hi), &mb.key_hi,
              (u32)(offsetof(r72tt_mailbox_t, keys_done) - offsetof(r72tt_mailbox_t, key_hi)));
  /* Make sure the job has landed before starting it */
  (void)tt_l1_read32(dev, tile, MB_FIELD(core, flags));
  tt_l1_write32(dev, tile, MB_FIELD(core, seq), seq);
}

static void read_job_result(tt_device_t *dev, const job_t *job, r72tt_mailbox_t *mb)
{
  tt_l1_read(dev, job->worker / R72TT_CORES, R72TT_MAILBOX(job->worker % R72TT_CORES),
             mb, sizeof(*mb));
}

static int job_result_consistent(const job_t *job, const r72tt_mailbox_t *mb)
{
  return mb->keys_done <= job->count && mb->cmc_count <= job->count &&
         (mb->found ? mb->keys_done < job->count : mb->keys_done == job->count) &&
         (!mb->cmc_count || mb->cmc_index <= mb->keys_done);
}

static int boot_firmware(tt_device_t *dev, int devicenum)
{
  r72tt_mailbox_t mb;
  unsigned t, c, workers = dev->tile_count * R72TT_CORES;
  u32 *all = device_state[devicenum].workers;

  if (r72tt_tile_fw_size > R72TT_FW_MAX_BYTES)
  {
    Log("Tenstorrent: firmware image is too large (%u bytes)\n", r72tt_tile_fw_size);
    return -1;
  }
  memset(&mb, 0, sizeof(mb));
  for (t = 0; t < dev->tile_count; t++)
  {
    tt_reg_write32(dev, t, TT_REG_SOFT_RESET, TT_SOFT_RESET_ALL);
    tt_l1_write(dev, t, 0, r72tt_tile_fw, r72tt_tile_fw_size);
    for (c = 0; c < R72TT_CORES; c++)
      tt_l1_write(dev, t, R72TT_MAILBOX(c), &mb, sizeof(mb));
    tt_reg_write32(dev, t, TT_REG_NCRISC_RESET_PC, R72TT_CORE_NCRISC * R72TT_ENTRY_STRIDE);
    tt_reg_write32(dev, t, TT_REG_NCRISC_RESET_PC_OVERRIDE, 1);
    for (c = 0; c < 3; c++)
      tt_reg_write32(dev, t, TT_REG_TRISC0_RESET_PC + 4 * c, (R72TT_CORE_TRISC0 + c) * R72TT_ENTRY_STRIDE);
    tt_reg_write32(dev, t, TT_REG_TRISC_RESET_PC_OVERRIDE, 7);
    /* Make sure the image and mailboxes have landed before the cores start */
    (void)tt_l1_read32(dev, t, R72TT_MAILBOX(R72TT_CORES - 1));
  }
  for (t = 0; t < dev->tile_count; t++)
    tt_reg_write32(dev, t, TT_REG_SOFT_RESET, TT_SOFT_RESET_ALL & ~(TT_SOFT_RESET_BRISC |
                   TT_SOFT_RESET_NCRISC | (7 * TT_SOFT_RESET_TRISC0)));

  for (c = 0; c < workers; c++)
    all[c] = c;
  if (wait_workers(dev, all, workers, offsetof(r72tt_mailbox_t, magic),
                   R72TT_MAGIC_READY, R72TT_BOOT_CLOCKS) < 0)
  {
    Log("Tenstorrent: firmware did not start on device %d\n", devicenum);
    return -1;
  }
  for (c = 0; c < workers; c++)
  {
    if (worker_read32(dev, c, offsetof(r72tt_mailbox_t, abi_version)) != R72TT_ABI_VERSION)
    {
      Log("Tenstorrent: firmware interface mismatch on tile %u core %u\n",
          c / R72TT_CORES, c % R72TT_CORES);
      return -1;
    }
  }
  device_state[devicenum].seq = 0;
  for (c = 0; c < CLASSES; c++)
    device_state[devicenum].clocks_per_key[c] = initial_clocks_per_key[c];
  return 0;
}

/*
 * Checks one class of worker on every tile before first use: the scalar
 * code on each of the five RISC-V cores, or every tile's SFPU or RVV
 * worker. With 'verify' that worker runs in verify mode, so that its result
 * for every key (RVV: every key in a whole block of 16) is also compared
 * against the scalar code.
 *
 * Each job must find a key planted 27 keys past a carry out of key byte
 * d - 1 into byte d (d = 1..8, in rotation; byte 0 is RC5_72UnitWork.L0.hi),
 * so the firmware's key increment is checked at every carry depth against
 * the host's key_add(). The other key bytes are pseudo-random. For the RVV
 * workers the planted key also moves from tile to tile, through all 16
 * positions in a block.
 */
static int self_check(tt_device_t *dev, int devicenum, int cls, int verify)
{
  job_t *jobs = device_state[devicenum].jobs;
  u32 *workers = device_state[devicenum].workers;
  const u32 count = 100, carry_at = 50;
  const unsigned n = cls != CLASS_SCALAR ? dev->tile_count : dev->tile_count * R72TT_CORES;
  const unsigned core = cls == CLASS_RVV ? R72TT_CORE_TRISC2 : R72TT_CORE_TRISC1;
  RC5_72UnitWork w;
  unsigned i, failed = 0;
  u32 seq = ++device_state[devicenum].seq;
  u32 seed = 0x12345678u;

  memset(&w, 0, sizeof(w));
  w.plain.lo = 0x6CF2BD15u;
  w.plain.hi = 0x989E1475u;
  for (i = 0; i < n; i++)
  {
    /* five cores per tile and eight depths: every core of a tile gets a
       different depth, and each kind of core gets all of them */
    const unsigned depth = 1 + i % 8;
    const u32 planted = 77 + (cls == CLASS_RVV ? i % 16 : 0);
    u32 bytes[9], hi, mid, lo, clo, chi;
    unsigned b;

    for (b = 0; b < 9; b++)
    {
      seed = seed * 1103515245u + 12345u;
      bytes[b] = (b == 0) ? 0xFFu - (carry_at - 1) : (b < depth) ? 0xFFu : seed >> 24;
    }
    jobs[i].worker = workers[i] = cls != CLASS_SCALAR ? i * R72TT_CORES + core : i;
    jobs[i].offset = 0;
    jobs[i].count = count;
    jobs[i].hi = bytes[0];
    jobs[i].mid = (bytes[1] << 24) | (bytes[2] << 16) | (bytes[3] << 8) | bytes[4];
    jobs[i].lo = (bytes[5] << 24) | (bytes[6] << 16) | (bytes[7] << 8) | bytes[8];
    hi = jobs[i].hi;
    mid = jobs[i].mid;
    lo = jobs[i].lo;
    key_add(&hi, &mid, &lo, planted);
    rc5_72_encrypt(hi, mid, lo, w.plain.lo, w.plain.hi, &clo, &chi);
    w.cypher.lo = clo;
    w.cypher.hi = chi;
    post_job(dev, &jobs[i], &w, class_flags[cls] | (verify ? R72TT_FLAG_VERIFY : 0), seq);
  }
  if (wait_workers(dev, workers, n, offsetof(r72tt_mailbox_t, done_seq), seq,
                   (ui64)count * R72TT_MAX_CLOCKS_PER_KEY * 4 + R72TT_BOOT_CLOCKS) < 0)
  {
    Log("Tenstorrent: self-check did not complete on device %d\n", devicenum);
    return -1;
  }
  for (i = 0; i < n; i++)
  {
    const u32 planted = 77 + (cls == CLASS_RVV ? i % 16 : 0);
    r72tt_mailbox_t mb;

    read_job_result(dev, &jobs[i], &mb);
    if (mb.verify_errors || !mb.found || mb.keys_done != planted ||
        !mb.cmc_count || mb.cmc_index != planted)
    {
      if (!failed++)
        Log("Tenstorrent: self-check failed on tile %u core %u: %u errors, found %u at %u\n",
            jobs[i].worker / R72TT_CORES, jobs[i].worker % R72TT_CORES,
            mb.verify_errors, mb.found, mb.keys_done);
    }
  }
  if (failed)
  {
    Log("Tenstorrent: %s self-check failed on %u of %u cores of device %d\n",
        class_name[cls], failed, n, devicenum);
    return -1;
  }
  return 0;
}

/* Checks 'count' keys from w->L0 on the device. On return, *keys_done is
   'count' or the index of the full match, and *cmc and *cmc_index describe
   the partial matches in [0, *keys_done]. Returns 1 for a full match, 0 if
   none, < 0 on device failure. */
static int run_keys(tt_device_t *dev, int devicenum, const RC5_72UnitWork *w, u32 count,
                    int mode, u32 *keys_done, u32 *cmc, u32 *cmc_index)
{
  device_state_t *ds = &device_state[devicenum];
  job_t *jobs = ds->jobs;
  u32 *job_workers = ds->workers;
  const unsigned workers = dev->tile_count * R72TT_CORES;
  double weight[CLASSES], total_weight = 0, cumulative = 0;
  ui64 class_keys[CLASSES] = { 0, 0, 0, 0 }, class_cycles[CLASSES] = { 0, 0, 0, 0 };
  unsigned i, n = 0;
  u32 offset = 0, seq;
  si64 clocks;
  int found = 0;

  /* Size every worker's range by its expected speed */
  for (i = 0; i < CLASSES; i++)
    weight[i] = 1.0 / ds->clocks_per_key[i];
  for (i = 0; i < R72TT_CORES; i++)
    total_weight += dev->tile_count * weight[worker_class(mode, i)];

  seq = ++ds->seq;
  for (i = 0; i < workers; i++)
  {
    const int cls = worker_class(mode, i % R72TT_CORES);
    u32 end;

    cumulative += weight[cls];
    end = (i + 1 == workers) ? count : (u32)((double)count * (cumulative / total_weight));
    if (end <= offset)
      continue;
    jobs[n].worker = job_workers[n] = i;
    jobs[n].offset = offset;
    jobs[n].count = end - offset;
    jobs[n].hi = w->L0.hi;
    jobs[n].mid = w->L0.mid;
    jobs[n].lo = w->L0.lo;
    key_add(&jobs[n].hi, &jobs[n].mid, &jobs[n].lo, offset);
    post_job(dev, &jobs[n], w, class_flags[cls], seq);
    offset = end;
    n++;
  }

  clocks = wait_workers(dev, job_workers, n, offsetof(r72tt_mailbox_t, done_seq), seq,
                        (ui64)count / workers * R72TT_MAX_CLOCKS_PER_KEY * R72TT_CORES +
                        R72TT_BOOT_CLOCKS);
  if (clocks < 0)
  {
    Log("Tenstorrent: device %d did not complete %u keys\n", devicenum, count);
    return -1;
  }

  /* Merge in key order: everything before the first full match counts */
  *cmc = 0;
  *cmc_index = 0;
  *keys_done = count;
  for (i = 0; i < n; i++)
  {
    const int cls = worker_class(mode, jobs[i].worker % R72TT_CORES);
    r72tt_mailbox_t mb;

    read_job_result(dev, &jobs[i], &mb);
    if (!job_result_consistent(&jobs[i], &mb))
    {
      Log("Tenstorrent: inconsistent results from tile %u core %u of device %d\n",
          jobs[i].worker / R72TT_CORES, jobs[i].worker % R72TT_CORES, devicenum);
      return -1;
    }
    if (mb.verify_errors)
    {
      /* a candidate of an assembly scan that the C check did not confirm */
      Log("Tenstorrent: key scan and check disagree on tile %u core %u of device %d\n",
          jobs[i].worker / R72TT_CORES, jobs[i].worker % R72TT_CORES, devicenum);
      return -1;
    }
    if (!mb.found)
    {
      class_keys[cls] += mb.keys_done;
      class_cycles[cls] += ((ui64)mb.cycles_hi << 32) | mb.cycles_lo;
    }
    if (found)
      continue;  /* only collecting statistics past the match */
    if (mb.cmc_count)
    {
      *cmc += mb.cmc_count;
      *cmc_index = jobs[i].offset + mb.cmc_index;
    }
    if (mb.found)
    {
      *keys_done = jobs[i].offset + mb.keys_done;
      found = 1;
    }
  }

  /* Track the speeds of the worker classes, from runs long enough to tell */
  for (i = 0; i < CLASSES; i++)
  {
    if (class_keys[i] >= 256u * dev->tile_count)
    {
      ds->clocks_per_key[i] = 0.5 * ds->clocks_per_key[i] +
                              0.5 * (double)class_cycles[i] / (double)class_keys[i];
    }
  }

  if (show_stats > 0)
  {
    /* kept short: screen output is cut at 80 columns */
    Log("TT: %u keys, %u jobs, clk/key %.1f/%.1f/%.1f\n", count, n,
        ds->clocks_per_key[CLASS_SCALAR],
        mode == MODE_RISCV ? 0.0 : ds->clocks_per_key[worker_class(mode, R72TT_CORE_TRISC1)],
        mode_uses_class(mode, CLASS_RVV) ? ds->clocks_per_key[CLASS_RVV] : 0.0);
  }
  return found;
}

static s32 rc5_72_unit_func_tt(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, int mode)
{
  const int devicenum = rc5_72unitwork->devicenum;
  device_state_t *ds;
  tt_device_t *dev;
  u32 keys_done = 0, cmc = 0, cmc_index = 0;
  int found, cls;

  if (show_stats < 0)
  {
    const char *env = getenv("DNETC_TT_STATS");
    show_stats = (env && *env && *env != '0') ? 1 : 0;
  }
  dev = tt_lock_device(devicenum);
  if (!dev)
  {
    RaiseExitRequestTrigger();
    return -1;
  }
  if (dev->is_sim && mode_uses_class(mode, CLASS_RVV))
  {
    /* core selection substitutes another core, so this is not reached */
    Log("Tenstorrent: the simulator does not have the RISC-V vector unit\n");
    tt_unlock_device(devicenum);
    RaiseExitRequestTrigger();
    return -1;
  }
  ds = &device_state[devicenum];
  if (!ds->booted)
  {
    if (boot_firmware(dev, devicenum) < 0)
    {
      tt_unlock_device(devicenum);
      RaiseExitRequestTrigger();
      return -1;
    }
    ds->booted = 1;
  }
  /* Every core also uses the scalar code, on the other RISC-V cores */
  for (cls = 0; cls < CLASSES; cls++)
  {
    if (ds->checked[cls] || !mode_uses_class(mode, cls))
      continue;
    /* The RVV worker scans differently in verify mode (one block per call
       of its key scan), so it is also checked as it normally runs */
    if (self_check(dev, devicenum, cls, cls != CLASS_SCALAR) < 0 ||
        (cls == CLASS_RVV && self_check(dev, devicenum, cls, 0) < 0))
    {
      tt_unlock_device(devicenum);
      RaiseExitRequestTrigger();
      return -1;
    }
    ds->checked[cls] = 1;
  }

  found = (*iterations == 0) ? 0 :
          run_keys(dev, devicenum, rc5_72unitwork, *iterations, mode, &keys_done, &cmc, &cmc_index);
  tt_unlock_device(devicenum);
  if (found < 0)
  {
    RaiseExitRequestTrigger();
    return -1;
  }
  if (*iterations == 0)
    return RESULT_NOTHING;

  if (cmc)
  {
    u32 hi = rc5_72unitwork->L0.hi, mid = rc5_72unitwork->L0.mid, lo = rc5_72unitwork->L0.lo;
    key_add(&hi, &mid, &lo, cmc_index);
    rc5_72unitwork->check.count += cmc;
    rc5_72unitwork->check.hi = hi;
    rc5_72unitwork->check.mid = mid;
    rc5_72unitwork->check.lo = lo;
  }

  if (found)
  {
    /* Confirm the match with the reference core before reporting it */
    RC5_72UnitWork t;
    u32 one = 1;

    memcpy(&t, rc5_72unitwork, sizeof(t));
    work_key_add(&t, keys_done);
    if (rc5_72_unit_func_ansi_1(&t, &one, NULL) != RESULT_FOUND)
    {
      Log("Tenstorrent: false positive from device %d at key %02x:%08x:%08x!\n",
          devicenum, t.L0.hi, t.L0.mid, t.L0.lo);
      RaiseExitRequestTrigger();
      return -1;
    }
    work_key_add(rc5_72unitwork, keys_done);
    *iterations = keys_done;
    return RESULT_FOUND;
  }

  work_key_add(rc5_72unitwork, *iterations);
  rc5_72unitwork->optimal_timeslice_increment = dev->tile_count * R72TT_CORES * 64;
  return RESULT_NOTHING;
}

s32 rc5_72_unit_func_tt_riscv(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  return rc5_72_unit_func_tt(rc5_72unitwork, iterations, MODE_RISCV);
}

s32 rc5_72_unit_func_tt_sfpu(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  return rc5_72_unit_func_tt(rc5_72unitwork, iterations, MODE_SFPU);
}

s32 rc5_72_unit_func_tt_sfpu_rvv(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  return rc5_72_unit_func_tt(rc5_72unitwork, iterations, MODE_SFPU_RVV);
}
