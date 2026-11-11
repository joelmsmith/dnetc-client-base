/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 key scan for the vector unit of RISC-V core T2 (TRISC2), the only
 * core of a Tensix tile that has one: part of the V extension, with 128-bit
 * registers and elements of at most 32 bits.
 *
 *   uint32_t r72tt_scan_rvv(const scan_t *ctx, uint32_t hi, uint32_t n,
 *                           uint32_t ea[16]);
 *
 * Checks the keys whose low byte is hi, hi + 1, ..., hi + n - 1 (hi and n
 * multiples of 16, n >= 16, hi + n <= 256), 16 at a time, all with the key
 * schedule prefix in ctx. Returns the index of the first block of 16 keys
 * that has a key whose encrypted low word equals cypher_lo, or n if there
 * is none; the caller finds the key with r72tt_scan(). ea[] receives the
 * encrypted low words of the last block checked.
 *
 * What shapes this code (measured on a Blackhole p150a):
 * - A vector instruction can start every cycle, but its result takes three
 *   cycles, and it also waits for the previous write to its destination
 *   register. So the 16 keys are two independent streams of 8 (two
 *   registers per value, LMUL = 2) with their instructions interleaved.
 * - There is no vector rotate. rotl(x, 3) is (x >> 29) + 8 * x; a rotate by
 *   a variable amount is two shifts, by s and by -s (shifts use the low five
 *   bits of the amount), and an OR.
 * - Each stream has eight registers: A, the three L words, eA, eB and two
 *   temporaries. S[] lives on the stack, in the core's local data RAM.
 * - An instruction the core does not implement is not trapped, so only
 *   instructions known to work are used: vadd, vxor, vor, vsll, vsrl,
 *   vrsub, vmacc, vid, vmv.v.x, vle32, vse32, vmseq, vmor, vfirst.
*/

#include "r72tt_scan.h"

.option arch, +zve32x

/* Vector register groups of stream 0 and stream 1 */
#define A_0    v0
#define X_0    v2    /* temporary: S[i] being built, then the rotate amount */
#define Y_0    v4    /* temporary: the left half of a rotate */
#define L0_0   v6
#define L1_0   v8
#define L2_0   v10
#define EA_0   v12
#define EB_0   v14
#define A_1    v16
#define X_1    v18
#define Y_1    v20
#define L0_1   v22
#define L1_1   v24
#define L2_1   v26
#define EA_1   v28
#define EB_1   v30

/* Scalar registers */
#define CTX    a0
#define HI     a1    /* low key byte of the first key of the block */
#define END    a2
#define EAOUT  a3
#define START  a4
#define P0     t0    /* S[i] of stream 0 */
#define P1     t1    /* S[i] of stream 1 */
#define PIQ    t2    /* first pass: P + i*Q */
#define CNT    t3
#define TMP    t4
#define QR     t5
#define EIGHT  t6

#define S_BYTES   64                 /* one S[i]: two streams of 8 keys */
#define FRAME     (26 * S_BYTES)

#define NEXT_S \
  addi  P0, P0, S_BYTES; \
  addi  P1, P1, S_BYTES;

/* With X = S[i] + A: A = S[i] = rotl3(X + L[p]) */
#define KS_A(p) \
  vadd.vv   X_0, X_0, L##p##_0; \
  vadd.vv   X_1, X_1, L##p##_1; \
  vsrl.vi   A_0, X_0, 29; \
  vsrl.vi   A_1, X_1, 29; \
  vmacc.vx  A_0, EIGHT, X_0; \
  vmacc.vx  A_1, EIGHT, X_1;

/* L[j] = rotl(L[j] + A + L[p], A + L[p]) */
#define KS_L(j, p) \
  vadd.vv   X_0, A_0, L##p##_0; \
  vadd.vv   X_1, A_1, L##p##_1; \
  vadd.vv   L##j##_0, L##j##_0, X_0; \
  vadd.vv   L##j##_1, L##j##_1, X_1; \
  vsll.vv   Y_0, L##j##_0, X_0; \
  vsll.vv   Y_1, L##j##_1, X_1; \
  vrsub.vi  X_0, X_0, 0; \
  vrsub.vi  X_1, X_1, 0; \
  vsrl.vv   L##j##_0, L##j##_0, X_0; \
  vsrl.vv   L##j##_1, L##j##_1, X_1; \
  vor.vv    L##j##_0, L##j##_0, Y_0; \
  vor.vv    L##j##_1, L##j##_1, Y_1;

/* Key schedule step updating L[j], with L[p] the L word updated before it.
   First pass: S[i] starts as P + i*Q and is stored */
#define STEP1(j, p) \
  vadd.vx   X_0, A_0, PIQ; \
  vadd.vx   X_1, A_1, PIQ; \
  add       PIQ, PIQ, QR; \
  KS_A(p) \
  vse32.v   A_0, (P0); \
  vse32.v   A_1, (P1); \
  NEXT_S \
  KS_L(j, p)

#define LOAD_S \
  vle32.v   X_0, (P0); \
  vle32.v   X_1, (P1); \
  vadd.vv   X_0, X_0, A_0; \
  vadd.vv   X_1, X_1, A_1;

/* Second pass: S[i] is loaded and stored back */
#define STEP2(j, p) \
  LOAD_S \
  KS_A(p) \
  vse32.v   A_0, (P0); \
  vse32.v   A_1, (P1); \
  NEXT_S \
  KS_L(j, p)

/* Third pass: S[i] is used by the encryption as soon as it is computed */
#define STEP3(j, p) \
  LOAD_S \
  KS_A(p) \
  NEXT_S \
  KS_L(j, p)

/* Encryption half round d = rotl(d ^ o, o) + A */
#define HALF(d, o) \
  vxor.vv   d##_0, d##_0, o##_0; \
  vxor.vv   d##_1, d##_1, o##_1; \
  vsll.vv   Y_0, d##_0, o##_0; \
  vsll.vv   Y_1, d##_1, o##_1; \
  vrsub.vi  X_0, o##_0, 0; \
  vrsub.vi  X_1, o##_1, 0; \
  vsrl.vv   d##_0, d##_0, X_0; \
  vsrl.vv   d##_1, d##_1, X_1; \
  vor.vv    d##_0, d##_0, Y_0; \
  vor.vv    d##_1, d##_1, Y_1; \
  vadd.vv   d##_0, d##_0, A_0; \
  vadd.vv   d##_1, d##_1, A_1;

.section .text
.global r72tt_scan_rvv
.balign 16
r72tt_scan_rvv:
  addi  sp, sp, -FRAME
  mv    START, HI
  add   END, HI, END
  li    QR, 0x9E3779B9
  li    EIGHT, 8
  vsetvli TMP, EIGHT, e32, m2, ta, ma

next_block:
  /* L[2] = rotl(hi + ab, ab) for hi = HI + lane; stream 1 has lanes 8..15 */
  lw    TMP, SCAN_AB(CTX)
  vid.v     X_0
  vid.v     X_1
  add   CNT, HI, TMP
  vadd.vx   X_0, X_0, CNT
  addi  CNT, CNT, 8
  vadd.vx   X_1, X_1, CNT
  vmv.v.x   Y_0, TMP
  vmv.v.x   Y_1, TMP
  vsll.vv   L2_0, X_0, Y_0
  vsll.vv   L2_1, X_1, Y_1
  vrsub.vi  Y_0, Y_0, 0
  vrsub.vi  Y_1, Y_1, 0
  vsrl.vv   X_0, X_0, Y_0
  vsrl.vv   X_1, X_1, Y_1
  vor.vv    L2_0, L2_0, X_0
  vor.vv    L2_1, L2_1, X_1

  /* S[0..2], L[0..1] and A = S[2] from the prefix */
  mv    P0, sp
  addi  P1, sp, S_BYTES / 2
  lw    TMP, SCAN_S0(CTX)
  vmv.v.x   X_0, TMP
  lw    TMP, SCAN_S1(CTX)
  vmv.v.x   X_1, TMP
  lw    TMP, SCAN_S2(CTX)
  vmv.v.x   A_0, TMP
  vmv.v.x   A_1, TMP
  vse32.v   X_0, (P0)
  vse32.v   X_0, (P1)
  NEXT_S
  vse32.v   X_1, (P0)
  vse32.v   X_1, (P1)
  NEXT_S
  vse32.v   A_0, (P0)
  vse32.v   A_1, (P1)
  NEXT_S
  lw    TMP, SCAN_L0(CTX)
  vmv.v.x   L0_0, TMP
  vmv.v.x   L0_1, TMP
  lw    TMP, SCAN_L1(CTX)
  vmv.v.x   L1_0, TMP
  vmv.v.x   L1_1, TMP

  /* First pass: steps 3..25, updating L[0], L[1], L[2], ... */
  li    PIQ, 0x9287BE8E         /* P + 3*Q */
  li    CNT, 8
  .balign 16
1:
  STEP1(0, 2)
  STEP1(1, 0)
  addi  CNT, CNT, -1
  beqz  CNT, 2f
  STEP1(2, 1)
  j     1b
2:

  /* Second pass: steps 0..25, updating L[2], L[0], L[1], ... */
  mv    P0, sp
  addi  P1, sp, S_BYTES / 2
  li    CNT, 9
  .balign 16
1:
  STEP2(2, 1)
  STEP2(0, 2)
  addi  CNT, CNT, -1
  beqz  CNT, 2f
  STEP2(1, 0)
  j     1b
2:

  /* Third pass fused with the encryption: steps 0..24, updating L[1],
     L[2], L[0], ...; S[25] and the last L update only feed eB */
  mv    P0, sp
  addi  P1, sp, S_BYTES / 2
  STEP3(1, 0)
  lw    TMP, SCAN_PLAIN_LO(CTX)
  vadd.vx   EA_0, A_0, TMP      /* eA = plain_lo + S[0] */
  vadd.vx   EA_1, A_1, TMP
  STEP3(2, 1)
  lw    TMP, SCAN_PLAIN_HI(CTX)
  vadd.vx   EB_0, A_0, TMP      /* eB = plain_hi + S[1] */
  vadd.vx   EB_1, A_1, TMP
  /* steps 2..23: three times six steps, then four */
  li    CNT, 4
  .balign 16
1:
  STEP3(0, 2) HALF(EA, EB)
  STEP3(1, 0) HALF(EB, EA)
  STEP3(2, 1) HALF(EA, EB)
  STEP3(0, 2) HALF(EB, EA)
  addi  CNT, CNT, -1
  beqz  CNT, 2f
  STEP3(1, 0) HALF(EA, EB)
  STEP3(2, 1) HALF(EB, EA)
  j     1b
2:
  LOAD_S
  KS_A(0)
  HALF(EA, EB)

  vse32.v   EA_0, (EAOUT)
  addi  TMP, EAOUT, 32
  vse32.v   EA_1, (TMP)
  lw    TMP, SCAN_CYPHER_LO(CTX)
  vmseq.vx  v0, EA_0, TMP
  vmseq.vx  v2, EA_1, TMP
  vmor.mm   v0, v0, v2
  vfirst.m  TMP, v0
  bgez  TMP, done
  addi  HI, HI, 16
  bne   HI, END, next_block

done:
  sub   a0, HI, START
  addi  sp, sp, FRAME
  ret
