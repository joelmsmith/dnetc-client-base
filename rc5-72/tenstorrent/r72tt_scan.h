/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Interface of the key scans in r72tt_scan.S and r72tt_scan_rvv.S, shared
 * by the firmware C code and the assembly.
*/

#ifndef R72TT_SCAN_H
#define R72TT_SCAN_H

/* scan_t offsets */
#define SCAN_S0         0
#define SCAN_S1         4
#define SCAN_S2         8
#define SCAN_L0        12
#define SCAN_L1        16
#define SCAN_AB        20
#define SCAN_PLAIN_LO  24
#define SCAN_PLAIN_HI  28
#define SCAN_CYPHER_LO 32

#ifndef __ASSEMBLER__
#include <stdint.h>

/* Key schedule state that depends only on (mid, lo) */
typedef struct
{
  uint32_t s0, s1, s2;  /* S[0..2] after the first pass */
  uint32_t l0, l1;      /* L[0..1] after the first pass */
  uint32_t ab;          /* A + B entering the L[2] update of step 2 */
} prefix_t;

typedef struct
{
  prefix_t p;
  uint32_t plain_lo, plain_hi, cypher_lo;
} scan_t;

/* Checks low key bytes hi .. hi + n - 1 (n >= 1, hi + n <= 256); returns
   the index of the first whose encrypted low word is cypher_lo, else n */
uint32_t r72tt_scan(const scan_t *ctx, uint32_t hi, uint32_t n);

/* TRISC2 only. Checks low key bytes hi .. hi + n - 1 in blocks of 16 (hi
   and n multiples of 16, n >= 16, hi + n <= 256); returns the index of the
   first block with a key whose encrypted low word is cypher_lo, else n.
   ea[] receives the encrypted low words of the last block checked. */
uint32_t r72tt_scan_rvv(const scan_t *ctx, uint32_t hi, uint32_t n, uint32_t ea[16]);

#endif /* __ASSEMBLER__ */
#endif /* R72TT_SCAN_H */
