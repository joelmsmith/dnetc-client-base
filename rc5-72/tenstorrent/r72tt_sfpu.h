/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 on the Tensix Vector Unit (SFPU), 32 keys at a time: one key per
 * lane, lane i checking low key byte 'base + i'. Driven by TRISC1, which
 * pushes the instruction stream to Tensix thread 1 and can read Dst back.
 *
 * The SFPU only computes the encrypted low word for every lane and records
 * which lanes match cypher_lo, accumulating one bit per lane and batch in
 * the ACC slot of Dst. The firmware then re-checks flagged keys (partial match
 * candidates, roughly one in 2^32) with the scalar core.
 *
 * Register use (LReg) during the key schedule and the encryption:
 *   0  A                  5  x (temp)
 *   1  eA                 6  y (temp)
 *   2  L[0]               7  eB
 *   3  L[1]
 *   4  L[2]
 *  12  cypher_lo (configured)                    15  2 * lane (fixed)
 *
 * The last key schedule pass is fused with the encryption, which uses S[i]
 * in the order that pass produces it. The key schedule steps are replayed
 * from the Tensix replay buffer, and the passes run as short loops whose
 * Tensix instructions are the same every iteration, so they stay in TRISC1's
 * small instruction cache: their Dst accesses advance through the S[] slots
 * using the Dst read/write counter (RWC), via the address modes below.
 *
 * Dst slots of 32 values (four rows, even columns), at address 4 * slot + RWC:
 *   0..25  S[]       26..51  P + i*Q       52  ACC (match bits)
 *   53     plain_lo  54      plain_hi      55  eA readback (verify mode)
 *   56     ACC bits OR'ed across each row of eight lanes
 *   57..62 the current key schedule prefix: ab, S[0..2], L[0..1]
*/

#ifndef R72TT_SFPU_H
#define R72TT_SFPU_H

#include <stdint.h>

/* --- Tensix instruction encodings ------------------------------------- */

#define TT_OP(op, params) ((uint32_t)(((uint32_t)(op) << 24) + (params)))

#define SFPLOAD_AM(d, am, addr)   TT_OP(0x70, ((d) << 20) + (SFP_INT32 << 16) + ((am) << 13) + (addr))
#define SFPSTORE_AM(d, am, addr)  TT_OP(0x72, ((d) << 20) + (SFP_INT32 << 16) + ((am) << 13) + (addr))
#define SFPLOAD(d, addr)          SFPLOAD_AM(d, AM_KEEP, addr)
#define SFPSTORE(d, addr)         SFPSTORE_AM(d, AM_KEEP, addr)
#define SFPLOADI(d, mod, imm16)   TT_OP(0x71, ((d) << 20) + ((mod) << 16) + ((imm16) & 0xFFFFu))
#define SFPOP(op, imm12, c, d, mod) \
  TT_OP(op, ((uint32_t)((imm12) & 0xFFF) << 12) + ((c) << 8) + ((d) << 4) + (mod))
#define SFPIADD(imm12, c, d, mod) SFPOP(0x79, imm12, c, d, mod)
#define SFPSHFT(imm12, c, d, mod) SFPOP(0x7A, imm12, c, d, mod)
#define SFPOR(imm12, c, d, mod)   SFPOP(0x7F, imm12, c, d, mod)
#define SFPLZ(c, d)               SFPOP(0x81, 0, c, d, 0)
#define SFPXOR(c, d)              SFPOP(0x8D, 0, c, d, 0)
#define SFPMOVNEG(c, d)           SFPOP(0x7C, 0, c, d, 1)   /* VD = VC ^ 2^31 */
#define SFPCONFIG(d)              TT_OP(0x91, ((d) << 4))
#define SFPSHFT2(imm12, c, d, mod) SFPOP(0x94, imm12, c, d, mod)
#define STALLWAIT(block, cond)    TT_OP(0xA2, ((block) << 15) + (cond))
#define SETDMAREG(value, reg16)   TT_OP(0x45, ((value) << 8) + (reg16))
#define SETC16(reg, value)        TT_OP(0xB2, ((reg) << 16) + (value))
#define SETRWC_DST0               TT_OP(0x37, 4)   /* Dst RWC = 0 */

/* Address modes (ADDR_MOD slots) for SFPLOAD/SFPSTORE: what happens to the
   Dst RWC after the access. Configured per thread with SETC16. */
#define AM_KEEP          0
#define AM_NEXT          1   /* advance one slot */
#define AM_CLEAR         2   /* back to 0 */
#define THCFG_ADDR_MOD_AB(am)    (12 + (am))
#define THCFG_ADDR_MOD_DST(am)   (28 + (am))
#define THCFG_ADDR_MOD_BIAS(am)  (47 + (am))
#define ADDR_MOD_DST_INCR(n)     (n)
#define ADDR_MOD_DST_CLEAR       (1 << 11)

#define SFP_INT32        4   /* SFPLOAD/SFPSTORE: 32-bit integer, no conversion */
#define SFPLOADI_LO16   10   /* write the low 16 bits, keep the high 16 */
#define SFPLOADI_HI16    8   /* write the high 16 bits, keep the low 16 */
#define SFPLOADI_U16     2   /* zero-extended 16-bit value */
#define IADD_REG         4   /* VD = VC + VD, flags untouched */
#define IADD_IMM         5   /* VD = VC + Imm12, flags untouched */
#define SHFT_REG         0   /* VD = VD << VC (VC < 0: logical >> -VC) */
#define SHFT_IMM_VC      5   /* VD = VC << Imm12 (Imm12 < 0: logical >> -Imm12) */
#define SHFT2_VB         5   /* VD = L[Imm12] << VC (VC < 0: logical >> -VC) */
#define SHFT2_ROR1       3   /* VD = VC with lanes rotated right by one within
                                each row of eight lanes */
#define LOGIC_VD         0   /* VD = VD op VC */
#define STALL_THCON      (1 << 5)    /* block ThCon (SETDMAREG) ... */
#define WAIT_SFPU        (1 << 11)   /* ... until the SFPU is idle */

#define LR_A    0
#define LR_EA   1
#define LR_L0   2
#define LR_L1   3
#define LR_L2   4
#define LR_X    5
#define LR_Y    6
#define LR_EB   7
#define LR_CLO  12
#define LR_LANE2 15

#define SLOT(n)         (4 * (n))
#define SLOT_S(i)       SLOT(i)
#define SLOT_SINIT(i)   SLOT(26 + (i))
#define SLOT_ACC        SLOT(52)
#define SLOT_PLAIN_LO   SLOT(53)
#define SLOT_PLAIN_HI   SLOT(54)
#define SLOT_EA         SLOT(55)
#define SLOT_ANY        SLOT(56)
#define SLOT_PAB        SLOT(57)
#define SLOT_PS(i)      SLOT(58 + (i))
#define SLOT_PL(i)      SLOT(61 + (i))

/* TRISC1 views */
#define TENSIX_INSTRN_BUF 0xFFE40000u
#define TENSIX_GPR63      (0xFFE00000u + 63 * 4)
#define TENSIX_GPR63_LO16 126
#define DST_BASE          0xFFBD8000u
#define DST_SLOT_LANE(slot_addr, lane) \
  (DST_BASE + ((((slot_addr) + (lane) / 8) * 16 + 2 * ((lane) & 7)) * 4))

/* Pushes a compile-time constant instruction as a single .ttinsn */
#define TTI_ROT2(w) ((((uint32_t)(w)) << 2) | (((uint32_t)(w)) >> 30))
#define TTI(w) __asm__ volatile(".word %0" :: "i"(TTI_ROT2(w)) : "memory")

/* Pushes an instruction computed at run time */
static inline void tt_push(uint32_t w)
{
  *(volatile uint32_t *)TENSIX_INSTRN_BUF = w;
}

static inline void sfpu_loadi32(uint32_t d, uint32_t v)
{
  tt_push(SFPLOADI(d, SFPLOADI_LO16, v & 0xFFFF));
  tt_push(SFPLOADI(d, SFPLOADI_HI16, v >> 16));
}

/* Waits for all SFPU instructions pushed so far to complete */
static void sfpu_wait(void)
{
  volatile uint32_t *done = (volatile uint32_t *)TENSIX_GPR63;
  *done = 0;
  while (*done)
    ;
  TTI(STALLWAIT(STALL_THCON, WAIT_SFPU));
  TTI(SETDMAREG(1, TENSIX_GPR63_LO16));
  while (!*done)
    ;
}

/* --- Replay buffer ---------------------------------------------------------- */

/* Tensix thread 1 keeps a 32-entry buffer of instructions that a single
   REPLAY expands to; a REPLAY occupies one slot of the instruction FIFO
   however many instructions it expands to. Replaying the key schedule steps
   keeps more SFPU work queued than the FIFO alone holds, so the SFPU does
   not run dry while TRISC1 is busy, and keeps TRISC1's code small. */
#define REPLAY(start, len, exec, load) \
  TT_OP(0x04, ((start) << 14) + ((len) << 4) + ((exec) << 1) + (load))

/* --- Key schedule -------------------------------------------------------- */

/* The A half of key schedule step i, with X = S[i] + A already formed by
   the step before: A = S[i] = rotl3(S[i] + A + L[j-1]); L[j-1] is the L word
   updated last */
#define V_A(Lp) \
  TTI(SFPIADD(0, Lp, LR_X, IADD_REG)); \
  TTI(SFPSHFT(3, LR_X, LR_Y, SHFT_IMM_VC)); \
  TTI(SFPSHFT(-29, LR_X, LR_A, SHFT_IMM_VC)); \
  TTI(SFPOR(0, LR_Y, LR_A, LOGIC_VD));

/* Rotates: SFPSHFT and SFPSHFT2 shift left by a non-negative amount and
   right by minus a negative one (both mod 32). s and s ^ 2^31 are equal
   mod 32 and of opposite signs, so shifting by both gives the two halves of
   rotl(x, s), whatever s is. */

/* The L half, once X = S[i+1] (or the initial S[i+1]) has been loaded:
   first X += A for the next step, which is A's last use, so A can then hold
   T = A + L[j-1] for L[j] = rotl(L[j] + T, T) without a copy. A is left
   holding junk; the next step's A half overwrites it before reading it. */
#define V_B_WORDS(Lj, Lp) \
  SFPIADD(0, LR_A, LR_X, IADD_REG), \
  SFPIADD(0, Lp, LR_A, IADD_REG), \
  SFPIADD(0, LR_A, Lj, IADD_REG), \
  SFPSHFT2(Lj, LR_A, LR_Y, SHFT2_VB), \
  SFPMOVNEG(LR_A, LR_A), \
  SFPSHFT(0, LR_A, Lj, SHFT_REG), \
  SFPOR(0, LR_Y, Lj, LOGIC_VD)

/* Encryption half round d = rotl(d ^ o, o) + A, with A = S[i] fresh from
   the third pass */
#define V_HALF_WORDS(d, o) \
  SFPXOR(o, d), \
  SFPSHFT2(d, o, LR_Y, SHFT2_VB), \
  SFPMOVNEG(o, LR_X), \
  SFPSHFT(0, LR_X, d, SHFT_REG), \
  SFPOR(0, LR_Y, d, LOGIC_VD), \
  SFPIADD(0, LR_A, d, IADD_REG)

/* Replay buffer contents (32 entries): the L half of a step for each L
   word, then the half round updating eA */
#define B_LEN       7
#define B_AT(j)     ((j) * B_LEN)
#define HALF_LEN    6
#define HALF_EA_AT  (3 * B_LEN)
#define REPLAY_LEN  (HALF_EA_AT + HALF_LEN)
static const uint32_t sfpu_replay[REPLAY_LEN] =
{
  V_B_WORDS(LR_L0, LR_L2),
  V_B_WORDS(LR_L1, LR_L0),
  V_B_WORDS(LR_L2, LR_L1),
  V_HALF_WORDS(LR_EA, LR_EB),
};

/* Key schedule step i updates A and L[j]. Each step loads the next step's
   S[i+1] between its A half and its L half. */
#define LPREV(j) ((j) == 0 ? LR_L2 : (j) == 1 ? LR_L0 : LR_L1)
#define V_B(j) TTI(REPLAY(B_AT(j), B_LEN, 0, 0));

/* Pass 1 and pass 2 steps: S[i] = A to Dst slot 'dst' (offset by the RWC)
   with address mode 'sam' updating the RWC, then the next S from slot 'src'
   with address mode 'lam' */
#define V_PASS(j, dst, sam, src, lam) \
  V_A(LPREV(j)) \
  TTI(SFPSTORE_AM(LR_A, sam, dst)); \
  TTI(SFPLOAD_AM(LR_X, lam, src)); \
  V_B(j)
#define V_PASS1(j) V_PASS(j, SLOT_S(3), AM_NEXT, SLOT_SINIT(3), AM_KEEP)
#define V_PASS2(j) V_PASS(j, SLOT_S(0), AM_NEXT, SLOT_S(0), AM_KEEP)

#define V_HALF_EA TTI(REPLAY(HALF_EA_AT, HALF_LEN, 0, 0));
#define V_HALF_EB \
  TTI(SFPXOR(LR_EA, LR_EB)); \
  TTI(SFPSHFT2(LR_EB, LR_EA, LR_Y, SHFT2_VB)); \
  TTI(SFPMOVNEG(LR_EA, LR_X)); \
  TTI(SFPSHFT(0, LR_X, LR_EB, SHFT_REG)); \
  TTI(SFPOR(0, LR_Y, LR_EB, LOGIC_VD)); \
  TTI(SFPIADD(0, LR_A, LR_EB, IADD_REG));

/* Third pass step i fused with the encryption: S[i] is used as soon as it is
   computed, so it is not stored. Half rounds alternate between eA and eB;
   they use X and Y as temporaries, so they come before the next load. */
#define V_PASS3_AM(j, half, am) \
  V_A(LPREV(j)) \
  half \
  TTI(SFPLOAD_AM(LR_X, am, SLOT_S(0))); \
  V_B(j)
#define V_PASS3(j, half) V_PASS3_AM(j, half, AM_NEXT)

/* Stores a zero ACC */
static inline void sfpu_clear_acc(void)
{
  TTI(SFPLOADI(LR_X, SFPLOADI_U16, 0));
  TTI(SFPSTORE(LR_X, SLOT_ACC));
}

/* One-time setup for a run: constants, the plaintext and the replay buffer */
static void sfpu_setup(uint32_t plain_lo, uint32_t plain_hi, uint32_t cypher_lo)
{
  uint32_t i;

  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_NEXT), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_NEXT), ADDR_MOD_DST_INCR(SLOT(1))));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_NEXT), 0));
  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_CLEAR), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_CLEAR), ADDR_MOD_DST_CLEAR));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_CLEAR), 0));
  /* Nothing has run on the SFPU yet in this run, so this cannot race */
  TTI(SETRWC_DST0);

  /* Recording with execution off: the recorded instructions must be pushed
     one store each, never as fused .ttinsn sequences */
  tt_push(REPLAY(0, REPLAY_LEN, 0, 1));
  for (i = 0; i < REPLAY_LEN; i++)
    tt_push(sfpu_replay[i]);

  sfpu_loadi32(0, cypher_lo);
  TTI(SFPCONFIG(LR_CLO));
  for (i = 0; i < 26; i++)
  {
    sfpu_loadi32(LR_X, 0xB7E15163u + i * 0x9E3779B9u);
    tt_push(SFPSTORE(LR_X, SLOT_SINIT(i)));
  }
  sfpu_loadi32(LR_X, plain_lo);
  TTI(SFPSTORE(LR_X, SLOT_PLAIN_LO));
  sfpu_loadi32(LR_X, plain_hi);
  TTI(SFPSTORE(LR_X, SLOT_PLAIN_HI));
  sfpu_clear_acc();
}

/* Stores a key schedule prefix for the batches that follow. Once per 256
   keys, so that each batch can load it with fixed instructions rather than
   TRISC1 encoding it into SFPLOADIs every time. */
static void sfpu_prefix(const prefix_t *p)
{
  sfpu_loadi32(LR_X, p->ab);
  TTI(SFPSTORE(LR_X, SLOT_PAB));
  sfpu_loadi32(LR_X, p->s0);
  TTI(SFPSTORE(LR_X, SLOT_PS(0)));
  sfpu_loadi32(LR_X, p->s1);
  TTI(SFPSTORE(LR_X, SLOT_PS(1)));
  sfpu_loadi32(LR_X, p->s2);
  TTI(SFPSTORE(LR_X, SLOT_PS(2)));
  sfpu_loadi32(LR_X, p->l0);
  TTI(SFPSTORE(LR_X, SLOT_PL(0)));
  sfpu_loadi32(LR_X, p->l1);
  TTI(SFPSTORE(LR_X, SLOT_PL(1)));
}

/*
 * Queues one batch: lanes check low key bytes base..base+31 for the
 * (mid, lo) whose key schedule prefix sfpu_prefix() stored last. Shifts the ACC slot left by
 * one and sets bit 0 of the lanes whose encrypted low word equals
 * cypher_lo. In verify mode the encrypted low words are also stored to
 * SLOT_EA.
 */
static void sfpu_batch(uint32_t base, int verify)
{
  uint32_t i;

  /* L[2] = rotl(hi + ab, ab), with hi = base + lane */
  TTI(SFPSHFT(-1, LR_LANE2, LR_L2, SHFT_IMM_VC));
  tt_push(SFPIADD(base, LR_L2, LR_L2, IADD_IMM));
  TTI(SFPLOAD(LR_X, SLOT_PAB));
  TTI(SFPIADD(0, LR_X, LR_L2, IADD_REG));
  TTI(SFPSHFT2(LR_L2, LR_X, LR_Y, SHFT2_VB));
  TTI(SFPMOVNEG(LR_X, LR_X));
  TTI(SFPSHFT(0, LR_X, LR_L2, SHFT_REG));
  TTI(SFPOR(0, LR_Y, LR_L2, LOGIC_VD));
  TTI(SFPLOAD(LR_A, SLOT_PS(2)));
  TTI(SFPLOAD(LR_L0, SLOT_PL(0)));
  TTI(SFPLOAD(LR_L1, SLOT_PL(1)));
  TTI(SFPLOAD(LR_X, SLOT_PS(0)));
  TTI(SFPSTORE(LR_X, SLOT_S(0)));
  TTI(SFPLOAD(LR_X, SLOT_PS(1)));
  TTI(SFPSTORE(LR_X, SLOT_S(1)));
  TTI(SFPSTORE(LR_A, SLOT_S(2)));
  /* eA and eB are free until the third pass */
  TTI(SFPLOAD(LR_EA, SLOT_PLAIN_LO));
  TTI(SFPLOAD(LR_EB, SLOT_PLAIN_HI));
  /* X = S[3] + A for the first step */
  TTI(SFPLOAD(LR_X, SLOT_SINIT(3)));
  TTI(SFPIADD(0, LR_A, LR_X, IADD_REG));

  /* First pass: steps 3..25, cycling L[0], L[1], L[2]; the last step loads
     S[0] for the second pass */
#pragma GCC unroll 1
  for (i = 0; i < 7; i++)
  {
    V_PASS1(0) V_PASS1(1) V_PASS1(2)
  }
  V_PASS1(0)
  V_PASS(1, SLOT_S(3), AM_CLEAR, SLOT_S(0), AM_KEEP)

  /* Second pass: steps 0..25, starting with L[2]; the last step loads S[0]
     for the third pass */
#pragma GCC unroll 1
  for (i = 0; i < 8; i++)
  {
    V_PASS2(2) V_PASS2(0) V_PASS2(1)
  }
  V_PASS2(2)
  V_PASS(0, SLOT_S(0), AM_CLEAR, SLOT_S(0), AM_NEXT)

  /* Third pass, starting with L[1], fused with the encryption:
     eA = plain_lo + S[0], eB = plain_hi + S[1], then the half rounds
     using S[2..24]. S[25] and the last L update only feed the encrypted
     high word, which the SFPU does not compute. */
  V_PASS3(1, TTI(SFPIADD(0, LR_A, LR_EA, IADD_REG));)
  V_PASS3(2, TTI(SFPIADD(0, LR_A, LR_EB, IADD_REG));)
#pragma GCC unroll 1
  for (i = 0; i < 3; i++)
  {
    V_PASS3(0, V_HALF_EA) V_PASS3(1, V_HALF_EB) V_PASS3(2, V_HALF_EA)
    V_PASS3(0, V_HALF_EB) V_PASS3(1, V_HALF_EA) V_PASS3(2, V_HALF_EB)
  }
  V_PASS3(0, V_HALF_EA) V_PASS3(1, V_HALF_EB) V_PASS3(2, V_HALF_EA)
  V_PASS3_AM(0, V_HALF_EB, AM_CLEAR)
  V_A(LPREV(1))
  V_HALF_EA

  if (verify)
    TTI(SFPSTORE(LR_EA, SLOT_EA));

  /* ACC = (ACC << 1) | (eA == cypher_lo), via clz(eA ^ cypher_lo) >> 5 */
  TTI(SFPXOR(LR_CLO, LR_EA));
  TTI(SFPLZ(LR_EA, LR_EA));
  TTI(SFPSHFT(-5, LR_EA, LR_EA, SHFT_IMM_VC));
  TTI(SFPLOAD(LR_X, SLOT_ACC));
  TTI(SFPSHFT(1, LR_X, LR_X, SHFT_IMM_VC));
  TTI(SFPOR(0, LR_EA, LR_X, LOGIC_VD));
  TTI(SFPSTORE(LR_X, SLOT_ACC));
}

/* Stores the OR of the ACC bits of each row of eight lanes to every lane of
   that row of SLOT_ANY, and waits until it (and ACC) can be read */
static void sfpu_flush_acc(void)
{
  TTI(SFPLOAD(LR_X, SLOT_ACC));
  TTI(SFPSHFT2(0, LR_X, LR_Y, SHFT2_ROR1));
  TTI(SFPOR(0, LR_Y, LR_X, LOGIC_VD));
  TTI(SFPSHFT2(0, LR_X, LR_Y, SHFT2_ROR1));
  TTI(SFPSHFT2(0, LR_Y, LR_Y, SHFT2_ROR1));
  TTI(SFPOR(0, LR_Y, LR_X, LOGIC_VD));
  TTI(SFPSHFT2(0, LR_X, LR_Y, SHFT2_ROR1));
  TTI(SFPSHFT2(0, LR_Y, LR_Y, SHFT2_ROR1));
  TTI(SFPSHFT2(0, LR_Y, LR_Y, SHFT2_ROR1));
  TTI(SFPSHFT2(0, LR_Y, LR_Y, SHFT2_ROR1));
  TTI(SFPOR(0, LR_Y, LR_X, LOGIC_VD));
  TTI(SFPSTORE(LR_X, SLOT_ANY));
  sfpu_wait();
}

#endif /* R72TT_SFPU_H */
