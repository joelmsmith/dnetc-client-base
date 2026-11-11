/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 scalar key scan for the Tensix RISC-V cores (RV32IM + Zbb).
 *
 *   uint32_t r72tt_scan(const scan_t *ctx, uint32_t hi, uint32_t n);
 *
 * Checks the keys whose low byte is hi, hi + 1, ..., hi + n - 1 (n >= 1,
 * hi + n <= 256), all with the key schedule prefix in ctx, and returns the
 * index of the first one whose encrypted low word equals cypher_lo, or n if
 * there is none. The caller rechecks such a candidate with rc5_72_check().
 *
 * Every S[] value, the three L words and one temporary live in registers
 * for the whole key, so a key needs no loads or stores of S[] and each key
 * schedule step is six instructions. A is always the S register written
 * last and B the L register written last, so neither needs a register of
 * its own. eA and eB share the registers of S[0] and S[1], which they
 * replace in the third pass once used; during the first pass the same
 * registers hold the running constant P + i*Q and Q.
*/

#include "r72tt_scan.h"

/* Registers: everything but zero and sp */
#define U    t0
#define L0   t1
#define L1   t2
#define L2   t3
#define S0   t4
#define S1   t5
#define S2   t6
#define S3   s0
#define S4   s1
#define S5   s2
#define S6   s3
#define S7   s4
#define S8   s5
#define S9   s6
#define S10  s7
#define S11  s8
#define S12  s9
#define S13  s10
#define S14  s11
#define S15  a0
#define S16  a1
#define S17  a2
#define S18  a3
#define S19  a4
#define S20  a5
#define S21  a6
#define S22  a7
#define S23  ra
#define S24  gp
#define S25  tp
#define EA   S0
#define EB   S1
#define PIQ  S0   /* first pass: P + i*Q */
#define QR   S1   /* first pass: Q */

/* Stack frame */
#define F_HI        0     /* low key byte of the key being checked */
#define F_END       4     /* hi + n */
#define F_START     8     /* the first hi */
#define F_AB       12     /* prefix, see prefix_t */
#define F_L0       16
#define F_L1       20
#define F_S2       24
#define F_S0       28
#define F_S1       32
#define F_PLAIN_LO 36
#define F_PLAIN_HI 40
#define F_CYPHER   44
#define F_P2Q      48     /* P + 2*Q */
#define F_Q        52
#define F_SAVE     56     /* ra, gp, tp, s0..s11 */
#define FRAME     128

/* Key schedule step i: S[i] = A = rotl3(S[i] + A + B), with sp = S[i-1]
   (A) and lp = L[j-1] (B); then L[j] = B = rotl(L[j] + A + B, A + B) */
#define STEP(si, sp, lj, lp) \
  add   si, si, sp; \
  add   si, si, lp; \
  rori  si, si, 29; \
  add   U, si, lp; \
  add   lj, lj, U; \
  rol   lj, lj, U;

/* The same in the first pass, where S[i] starts as P + i*Q */
#define STEP1(si, sp, lj, lp) \
  add   PIQ, PIQ, QR; \
  add   si, PIQ, sp; \
  add   si, si, lp; \
  rori  si, si, 29; \
  add   U, si, lp; \
  add   lj, lj, U; \
  rol   lj, lj, U;

/* Encryption half rounds with A = s */
#define HALF_A(s) \
  xor   EA, EA, EB; \
  rol   EA, EA, EB; \
  add   EA, EA, s;
#define HALF_B(s) \
  xor   EB, EB, EA; \
  rol   EB, EB, EA; \
  add   EB, EB, s;

#define SAVE(r, i)  sw r, F_SAVE + 4 * (i)(sp)
#define LOAD(r, i)  lw r, F_SAVE + 4 * (i)(sp)

.section .text
.global r72tt_scan
.balign 4
r72tt_scan:
  addi  sp, sp, -FRAME
  SAVE(ra, 0); SAVE(gp, 1); SAVE(tp, 2)
  SAVE(s0, 3); SAVE(s1, 4); SAVE(s2, 5); SAVE(s3, 6); SAVE(s4, 7); SAVE(s5, 8)
  SAVE(s6, 9); SAVE(s7, 10); SAVE(s8, 11); SAVE(s9, 12); SAVE(s10, 13); SAVE(s11, 14)
  sw    a1, F_HI(sp)
  sw    a1, F_START(sp)
  add   a2, a1, a2
  sw    a2, F_END(sp)
  lw    t0, SCAN_S0(a0)
  lw    t1, SCAN_S1(a0)
  lw    t2, SCAN_S2(a0)
  lw    t3, SCAN_L0(a0)
  lw    t4, SCAN_L1(a0)
  lw    t5, SCAN_AB(a0)
  sw    t0, F_S0(sp)
  sw    t1, F_S1(sp)
  sw    t2, F_S2(sp)
  sw    t3, F_L0(sp)
  sw    t4, F_L1(sp)
  sw    t5, F_AB(sp)
  lw    t0, SCAN_PLAIN_LO(a0)
  lw    t1, SCAN_PLAIN_HI(a0)
  lw    t2, SCAN_CYPHER_LO(a0)
  li    t3, 0xF45044D5          /* P + 2*Q */
  li    t4, 0x9E3779B9          /* Q */
  sw    t0, F_PLAIN_LO(sp)
  sw    t1, F_PLAIN_HI(sp)
  sw    t2, F_CYPHER(sp)
  sw    t3, F_P2Q(sp)
  sw    t4, F_Q(sp)

next_key:
  /* L[2] = B = rotl(hi + ab, ab); A = S[2] */
  lw    U, F_AB(sp)
  lw    L2, F_HI(sp)
  lw    L0, F_L0(sp)
  lw    L1, F_L1(sp)
  lw    S2, F_S2(sp)
  lw    PIQ, F_P2Q(sp)
  lw    QR, F_Q(sp)
  add   L2, L2, U
  rol   L2, L2, U

  /* First pass: steps 3..25 */
  STEP1(S3, S2, L0, L2)
  STEP1(S4, S3, L1, L0)
  STEP1(S5, S4, L2, L1)
  STEP1(S6, S5, L0, L2)
  STEP1(S7, S6, L1, L0)
  STEP1(S8, S7, L2, L1)
  STEP1(S9, S8, L0, L2)
  STEP1(S10, S9, L1, L0)
  STEP1(S11, S10, L2, L1)
  STEP1(S12, S11, L0, L2)
  STEP1(S13, S12, L1, L0)
  STEP1(S14, S13, L2, L1)
  STEP1(S15, S14, L0, L2)
  STEP1(S16, S15, L1, L0)
  STEP1(S17, S16, L2, L1)
  STEP1(S18, S17, L0, L2)
  STEP1(S19, S18, L1, L0)
  STEP1(S20, S19, L2, L1)
  STEP1(S21, S20, L0, L2)
  STEP1(S22, S21, L1, L0)
  STEP1(S23, S22, L2, L1)
  STEP1(S24, S23, L0, L2)
  STEP1(S25, S24, L1, L0)

  /* Second pass: steps 0..25 */
  lw    S0, F_S0(sp)
  lw    S1, F_S1(sp)
  STEP(S0, S25, L2, L1)
  STEP(S1, S0, L0, L2)
  STEP(S2, S1, L1, L0)
  STEP(S3, S2, L2, L1)
  STEP(S4, S3, L0, L2)
  STEP(S5, S4, L1, L0)
  STEP(S6, S5, L2, L1)
  STEP(S7, S6, L0, L2)
  STEP(S8, S7, L1, L0)
  STEP(S9, S8, L2, L1)
  STEP(S10, S9, L0, L2)
  STEP(S11, S10, L1, L0)
  STEP(S12, S11, L2, L1)
  STEP(S13, S12, L0, L2)
  STEP(S14, S13, L1, L0)
  STEP(S15, S14, L2, L1)
  STEP(S16, S15, L0, L2)
  STEP(S17, S16, L1, L0)
  STEP(S18, S17, L2, L1)
  STEP(S19, S18, L0, L2)
  STEP(S20, S19, L1, L0)
  STEP(S21, S20, L2, L1)
  STEP(S22, S21, L0, L2)
  STEP(S23, S22, L1, L0)
  STEP(S24, S23, L2, L1)
  STEP(S25, S24, L0, L2)

  /* Third pass fused with the encryption: steps 0..24 and the half rounds
     with S[2..24]; S[25] and the last L update only feed eB */
  STEP(S0, S25, L1, L0)
  STEP(S1, S0, L2, L1)
  lw    U, F_PLAIN_LO(sp)
  add   EA, EA, U              /* eA = plain_lo + S[0] */
  STEP(S2, S1, L0, L2)
  lw    U, F_PLAIN_HI(sp)
  add   EB, EB, U              /* eB = plain_hi + S[1] */
  HALF_A(S2)
  STEP(S3, S2, L1, L0) HALF_B(S3)
  STEP(S4, S3, L2, L1) HALF_A(S4)
  STEP(S5, S4, L0, L2) HALF_B(S5)
  STEP(S6, S5, L1, L0) HALF_A(S6)
  STEP(S7, S6, L2, L1) HALF_B(S7)
  STEP(S8, S7, L0, L2) HALF_A(S8)
  STEP(S9, S8, L1, L0) HALF_B(S9)
  STEP(S10, S9, L2, L1) HALF_A(S10)
  STEP(S11, S10, L0, L2) HALF_B(S11)
  STEP(S12, S11, L1, L0) HALF_A(S12)
  STEP(S13, S12, L2, L1) HALF_B(S13)
  STEP(S14, S13, L0, L2) HALF_A(S14)
  STEP(S15, S14, L1, L0) HALF_B(S15)
  STEP(S16, S15, L2, L1) HALF_A(S16)
  STEP(S17, S16, L0, L2) HALF_B(S17)
  STEP(S18, S17, L1, L0) HALF_A(S18)
  STEP(S19, S18, L2, L1) HALF_B(S19)
  STEP(S20, S19, L0, L2) HALF_A(S20)
  STEP(S21, S20, L1, L0) HALF_B(S21)
  STEP(S22, S21, L2, L1) HALF_A(S22)
  STEP(S23, S22, L0, L2) HALF_B(S23)
  add   S24, S24, S23
  add   S24, S24, L0
  rori  S24, S24, 29
  HALF_A(S24)

  lw    U, F_CYPHER(sp)
  beq   EA, U, done
  lw    U, F_HI(sp)
  lw    EB, F_END(sp)
  addi  U, U, 1
  sw    U, F_HI(sp)
  bne   U, EB, next_key

done:
  lw    U, F_HI(sp)
  lw    EB, F_START(sp)
  sub   a0, U, EB
  LOAD(ra, 0); LOAD(gp, 1); LOAD(tp, 2)
  LOAD(s0, 3); LOAD(s1, 4); LOAD(s2, 5); LOAD(s3, 6); LOAD(s4, 7); LOAD(s5, 8)
  LOAD(s6, 9); LOAD(s7, 10); LOAD(s8, 11); LOAD(s9, 12); LOAD(s10, 13); LOAD(s11, 14)
  addi  sp, sp, FRAME
  ret
