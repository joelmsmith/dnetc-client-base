/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 key search firmware for a Tenstorrent Tensix tile. All five RISC-V
 * cores run this code; TRISC1 can instead drive the Vector Unit (see
 * r72tt_sfpu.h), and TRISC2 can use its RISC-V vector unit (see
 * r72tt_scan_rvv.S). See r72tt_fw.h for the host interface.
 *
 * The scalar core (r72tt_scan.S) checks one key at a time, fully unrolled,
 * and rc5_72_check() below rechecks its candidates. The key's low
 * byte ('hi') changes fastest, and the first key schedule steps don't depend
 * on it, so they are computed once per 256 keys. The last key schedule pass
 * is fused with the encryption, which needs S[] in the order that pass
 * produces it.
*/

#include <stddef.h>
#include <stdint.h>
#include "r72tt_fw.h"
#include "r72tt_scan.h"

#define P 0xB7E15163u
#define Q 0x9E3779B9u

/* r72tt_scan.S hard-codes these */
_Static_assert(P + 2 * Q == 0xF45044D5u, "P + 2Q");
_Static_assert(offsetof(scan_t, p.s0) == SCAN_S0 && offsetof(scan_t, p.s1) == SCAN_S1 &&
               offsetof(scan_t, p.s2) == SCAN_S2 && offsetof(scan_t, p.l0) == SCAN_L0 &&
               offsetof(scan_t, p.l1) == SCAN_L1 && offsetof(scan_t, p.ab) == SCAN_AB &&
               offsetof(scan_t, plain_lo) == SCAN_PLAIN_LO &&
               offsetof(scan_t, plain_hi) == SCAN_PLAIN_HI &&
               offsetof(scan_t, cypher_lo) == SCAN_CYPHER_LO, "scan_t layout");

#define REG_WALL_CLOCK_LO 0xFFB121F0u
#define REG_WALL_CLOCK_HI 0xFFB121F8u   /* latched by reading the low word */

#define RD32(addr) (*(volatile uint32_t *)(addr))
#define FENCE() __asm__ volatile("fence rw, rw" ::: "memory")

static inline uint32_t rotl(uint32_t x, uint32_t s)
{
  s &= 31;
  return (x << s) | (x >> ((32 - s) & 31));
}

static inline uint64_t wall_clock(void)
{
  uint32_t lo = RD32(REG_WALL_CLOCK_LO);
  uint32_t hi = RD32(REG_WALL_CLOCK_HI);
  return ((uint64_t)hi << 32) | lo;
}

/* The client's key increment (see r72ansi1.cpp) for a carry out of the low
   byte: mid and lo hold the remaining key bytes in reversed order. */
static void key_carry(uint32_t *mid, uint32_t *lo)
{
  uint32_t m = *mid + 0x01000000u;
  if (!(m & 0xFF000000u))
  {
    m = (m + 0x00010000u) & 0x00FFFFFFu;
    if (!(m & 0x00FF0000u))
    {
      m = (m + 0x00000100u) & 0x0000FFFFu;
      if (!(m & 0x0000FF00u))
      {
        m = (m + 0x00000001u) & 0x000000FFu;
        if (!m)
        {
          uint32_t l = *lo + 0x01000000u;
          if (!(l & 0xFF000000u))
          {
            l = (l + 0x00010000u) & 0x00FFFFFFu;
            if (!(l & 0x00FF0000u))
            {
              l = (l + 0x00000100u) & 0x0000FFFFu;
              if (!(l & 0x0000FF00u))
                l = (l + 0x00000001u) & 0x000000FFu;
            }
          }
          *lo = l;
        }
      }
    }
  }
  *mid = m;
}

static void key_prefix(uint32_t mid, uint32_t lo, prefix_t *p)
{
  uint32_t A, B;

  A = rotl(P, 3);
  B = rotl(lo + A, A);
  p->s0 = A;
  p->l0 = B;
  A = rotl(P + Q + A + B, 3);
  B = rotl(mid + A + B, A + B);
  p->s1 = A;
  p->l1 = B;
  A = rotl(P + 2 * Q + A + B, 3);
  p->s2 = A;
  p->ab = A + B;
}

/* One key schedule step: S[i] from 'si', then L[j] */
#define KS(si, Lj) \
  A = rotl((si) + A + B, 3); \
  B = Lj = rotl(Lj + A + B, A + B);

#define PASS1(i, Lj)  KS(P + (i) * Q, Lj) S[i] = A;
#define PASS2(i, Lj)  KS(S[i], Lj) S[i] = A;
#define PASS3(i, Lj)  KS(S[i], Lj)

/* Two half rounds per pair of S values, fused with the last pass */
#define ROUND(i, Lj, Lk) \
  PASS3(2 * (i), Lj) eA = rotl(eA ^ eB, eB) + A; \
  PASS3(2 * (i) + 1, Lk) eB = rotl(eB ^ eA, eA) + A;

/* Encrypts the plaintext with the key (hi, prefix). Sets *a to the
   encrypted plain_lo, and returns nonzero if that equals cypher_lo; then
   *b is the encrypted plain_hi. The reference for r72tt_scan(),
   r72tt_scan_rvv() and the SFPU, which only find candidates; this rechecks
   them. */
static inline int rc5_72_check(const prefix_t *p, uint32_t hi,
                               uint32_t plain_lo, uint32_t plain_hi,
                               uint32_t cypher_lo, uint32_t *a, uint32_t *b)
{
  uint32_t S[26];
  uint32_t L0 = p->l0, L1 = p->l1, L2;
  uint32_t A = p->s2, B, eA, eB;

  B = L2 = rotl(hi + p->ab, p->ab);
  S[0] = p->s0;
  S[1] = p->s1;
  S[2] = p->s2;

  PASS1( 3, L0) PASS1( 4, L1) PASS1( 5, L2)
  PASS1( 6, L0) PASS1( 7, L1) PASS1( 8, L2)
  PASS1( 9, L0) PASS1(10, L1) PASS1(11, L2)
  PASS1(12, L0) PASS1(13, L1) PASS1(14, L2)
  PASS1(15, L0) PASS1(16, L1) PASS1(17, L2)
  PASS1(18, L0) PASS1(19, L1) PASS1(20, L2)
  PASS1(21, L0) PASS1(22, L1) PASS1(23, L2)
  PASS1(24, L0) PASS1(25, L1)

  PASS2( 0, L2) PASS2( 1, L0) PASS2( 2, L1)
  PASS2( 3, L2) PASS2( 4, L0) PASS2( 5, L1)
  PASS2( 6, L2) PASS2( 7, L0) PASS2( 8, L1)
  PASS2( 9, L2) PASS2(10, L0) PASS2(11, L1)
  PASS2(12, L2) PASS2(13, L0) PASS2(14, L1)
  PASS2(15, L2) PASS2(16, L0) PASS2(17, L1)
  PASS2(18, L2) PASS2(19, L0) PASS2(20, L1)
  PASS2(21, L2) PASS2(22, L0) PASS2(23, L1)
  PASS2(24, L2) PASS2(25, L0)

  PASS3(0, L1) eA = plain_lo + A;
  PASS3(1, L2) eB = plain_hi + A;
  ROUND( 1, L0, L1) ROUND( 2, L2, L0) ROUND( 3, L1, L2)
  ROUND( 4, L0, L1) ROUND( 5, L2, L0) ROUND( 6, L1, L2)
  ROUND( 7, L0, L1) ROUND( 8, L2, L0) ROUND( 9, L1, L2)
  ROUND(10, L0, L1) ROUND(11, L2, L0)
  PASS3(24, L1) eA = rotl(eA ^ eB, eB) + A;
  *a = eA;
  if (eA != cypher_lo)
    return 0;
  PASS3(25, L2) eB = rotl(eB ^ eA, eA) + A;
  *b = eB;
  return 1;
}

#include "r72tt_sfpu.h"

/* Checks keys one at a time with r72tt_scan(). With 'rvv' (TRISC2 only),
   whole blocks of 16 low key bytes go to r72tt_scan_rvv() instead; a block
   it reports a candidate in is then scanned again by r72tt_scan(), as are
   the keys before the first and after the last whole block. */
static void run_scalar(volatile r72tt_mailbox_t *mb, int rvv)
{
  uint32_t hi = mb->key_hi, mid = mb->key_mid, lo = mb->key_lo;
  const uint32_t count = mb->count;
  const uint32_t plain_lo = mb->plain_lo, plain_hi = mb->plain_hi;
  const uint32_t cypher_lo = mb->cypher_lo, cypher_hi = mb->cypher_hi;
  const int verify = rvv && (mb->flags & R72TT_FLAG_VERIFY);
  uint32_t cmc_count = 0, cmc_index = 0, found = 0, errors = 0;
  uint32_t n, rescan = 0, rescan_hit = 0, ea[16];
  uint64_t start = wall_clock(), cycles;
  scan_t ctx;

  ctx.plain_lo = plain_lo;
  ctx.plain_hi = plain_hi;
  ctx.cypher_lo = cypher_lo;
  key_prefix(mid, lo, &ctx.p);
  for (n = 0; n < count; )
  {
    /* Scan to the end of the low byte's range or of the work, stopping at
       a candidate */
    uint32_t run = 256 - hi, k;
    if (run > count - n)
      run = count - n;
    if (rvv && !rescan && !(hi & 15) && run >= 16)
    {
      run = verify ? 16 : run & ~15u;
      k = r72tt_scan_rvv(&ctx, hi, run, ea);
      if (verify)
      {
        uint32_t lane, a, b, hit = 0;
        for (lane = 0; lane < 16; lane++)
        {
          hit |= rc5_72_check(&ctx.p, hi + lane, plain_lo, plain_hi, cypher_lo, &a, &b);
          if (ea[lane] != a)
            errors++;
        }
        if (hit != (k < run))
          errors++;   /* a candidate missed, or reported without one */
      }
      if (k < run)
      {
        /* a candidate among the next 16 keys */
        rescan = 16;
        rescan_hit = 0;
      }
    }
    else
    {
      if (rescan)
      {
        if (run > rescan)
          run = rescan;
      }
      else if (rvv && run > 16 - (hi & 15))
        run = 16 - (hi & 15);   /* up to the next whole block */
      k = r72tt_scan(&ctx, hi, run);
      if (k < run)
      {
        uint32_t a, b;
        rescan_hit = 1;
        if (rc5_72_check(&ctx.p, hi + k, plain_lo, plain_hi, cypher_lo, &a, &b))
        {
          cmc_count++;
          cmc_index = n + k;
          if (b == cypher_hi)
          {
            n += k;
            found = 1;
            break;
          }
        }
        else
          errors++;   /* r72tt_scan() and rc5_72_check() disagree */
        k++;
      }
      if (rescan)
      {
        rescan -= k;
        if (!rescan && !rescan_hit)
          errors++;   /* r72tt_scan_rvv() and r72tt_scan() disagree */
      }
    }
    n += k;
    hi += k;
    if (hi == 256)
    {
      hi = 0;
      key_carry(&mid, &lo);
      key_prefix(mid, lo, &ctx.p);
    }
  }

  cycles = wall_clock() - start;
  mb->keys_done = n;
  mb->cmc_count = cmc_count;
  mb->cmc_index = cmc_index;
  mb->found = found;
  mb->cycles_lo = (uint32_t)cycles;
  mb->cycles_hi = (uint32_t)(cycles >> 32);
  mb->verify_errors = errors;
}

/* Where the keys of a queued SFPU batch come from */
typedef struct
{
  uint32_t base, mid, lo;
} batch_t;

/* The Vector Unit version of run_scalar(), for TRISC1 only. Batch b covers
   low key bytes [base_b, base_b + 32), so key index 0 is lane 'first' of
   batch 0. Lanes outside the range are computed but ignored. */
static void run_sfpu(volatile r72tt_mailbox_t *mb)
{
  uint32_t hi = mb->key_hi, mid = mb->key_mid, lo = mb->key_lo;
  const uint32_t count = mb->count;
  const uint32_t plain_lo = mb->plain_lo, plain_hi = mb->plain_hi;
  const uint32_t cypher_lo = mb->cypher_lo, cypher_hi = mb->cypher_hi;
  const int verify = (mb->flags & R72TT_FLAG_VERIFY) != 0;
  const uint32_t first = hi & 31;
  const uint32_t batches = (first + count + 31) / 32;
  uint32_t cmc_count = 0, cmc_index = 0, found = 0, keys_done = count, errors = 0;
  uint32_t base = hi - first, b, group = 0, pending = 0;
  uint64_t start = wall_clock(), cycles;
  batch_t queued[32];
  prefix_t prefix;

  sfpu_setup(plain_lo, plain_hi, cypher_lo);
  key_prefix(mid, lo, &prefix);
  sfpu_prefix(&prefix);
  for (b = 0; b < batches && !found; b++)
  {
    queued[pending].base = base;
    queued[pending].mid = mid;
    queued[pending].lo = lo;
    pending++;
    sfpu_batch(base, verify);

    if (verify)
    {
      uint32_t lane;
      sfpu_wait();
      for (lane = 0; lane < 32; lane++)
      {
        uint32_t idx = b * 32 + lane - first, a, eb;
        if (idx < count)
        {
          rc5_72_check(&prefix, base + lane, plain_lo, plain_hi, cypher_lo, &a, &eb);
          if (*(volatile uint32_t *)DST_SLOT_LANE(SLOT_EA, lane) != a)
            errors++;
        }
      }
    }

    base += 32;
    if (base == 256)
    {
      base = 0;
      key_carry(&mid, &lo);
      key_prefix(mid, lo, &prefix);
      sfpu_prefix(&prefix);
    }

    if (pending == 32 || b + 1 == batches)
    {
      uint32_t j, lane, bits[32], any = 0;
      sfpu_flush_acc();
      /* Dst reads are slow on silicon: read the OR of each row of eight
         lanes, then each lane's bits once if any is set */
      for (lane = 0; lane < 32; lane += 8)
        any |= *(volatile uint32_t *)DST_SLOT_LANE(SLOT_ANY, lane);
      for (lane = 0; lane < 32 && any; lane++)
        bits[lane] = *(volatile uint32_t *)DST_SLOT_LANE(SLOT_ACC, lane);
      sfpu_clear_acc();
      /* batch j of the group holds bit (pending - 1 - j); scan in key order */
      for (j = 0; j < pending && any && !found; j++)
      {
        for (lane = 0; lane < 32; lane++)
        {
          uint32_t idx = (group + j) * 32 + lane - first, a, eb;
          prefix_t p;

          if (!((bits[lane] >> (pending - 1 - j)) & 1) || idx >= count)
            continue;
          key_prefix(queued[j].mid, queued[j].lo, &p);
          if (rc5_72_check(&p, queued[j].base + lane, plain_lo, plain_hi, cypher_lo, &a, &eb))
          {
            cmc_count++;
            cmc_index = idx;
            if (eb == cypher_hi)
            {
              found = 1;
              keys_done = idx;
              break;
            }
          }
        }
      }
      group += pending;
      pending = 0;
    }
  }

  cycles = wall_clock() - start;
  mb->keys_done = keys_done;
  mb->cmc_count = cmc_count;
  mb->cmc_index = cmc_index;
  mb->found = found;
  mb->cycles_lo = (uint32_t)cycles;
  mb->cycles_hi = (uint32_t)(cycles >> 32);
  mb->verify_errors = errors;
}

void r72tt_main(uint32_t core)
{
  volatile r72tt_mailbox_t *mb = (volatile r72tt_mailbox_t *)R72TT_MAILBOX(core);
  uint32_t done = mb->seq;

  mb->done_seq = done;
  mb->abi_version = R72TT_ABI_VERSION;
  FENCE();
  mb->magic = R72TT_MAGIC_READY;

  for (;;)
  {
    uint32_t seq;
    do
    {
      FENCE();
      seq = mb->seq;
    } while (seq == done);

    if (core == R72TT_CORE_TRISC1 && (mb->flags & R72TT_FLAG_SFPU))
      run_sfpu(mb);
    else
      run_scalar(mb, core == R72TT_CORE_TRISC2 && (mb->flags & R72TT_FLAG_RVV));
    FENCE();
    mb->done_seq = done = seq;
  }
}
