#!/usr/bin/env python3
#
# Copyright distributed.net 2026 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Generates r72tt_sfpu_lm.h: the RC5-72 batch for the Tensix Vector Unit
# (SFPU) that uses SFPLOADMACRO, and checks it first against a model of the
# documented SFPLOADMACRO behaviour and a plain RC5-72.
#
#   gen_sfpu_lm.py [-n KEYBATCHES] > ../r72tt_sfpu_lm.h
#
# SFPLOADMACRO loads a register and schedules instructions that run later in
# the same cycle as an instruction issued then, on another sub-unit of the
# SFPU. Here every load is an SFPLOADMACRO, and:
# - the store of S[i] is scheduled, so it takes no cycle of its own;
# - each SFPLOADMACRO's cycle also runs one add or sign flip scheduled by
#   the SFPLOADMACRO before it.
# A key schedule step then takes 11 cycles and an encryption half round 6.
# See the comments in the generated file.

import random
import sys

M32 = 0xFFFFFFFF
P, Q = 0xB7E15163, 0x9E3779B9

# LRegs
A, X0, X1, G, Y, L0, L1, L2 = 0, 1, 2, 3, 4, 5, 6, 7
CLO, LANE2 = 12, 15
REG_NAME = {0: 'LM_A', 1: 'LM_X0', 2: 'LM_X1', 3: 'LM_G', 4: 'LM_Y', 5: 'LM_L0', 6: 'LM_L1',
            7: 'LM_L2', 12: 'LR_CLO', 15: 'LR_LANE2'}
LREG = [L0, L1, L2]

# Dst slots (of four rows). A slot has even and odd columns: with
# DEST_WR_COL_EXCHANGE set every store goes to the odd ones, so the
# constants in the even ones survive the stores to their slots.
S_ACC, S_PLO, S_PHI, S_EA = 52, 53, 54, 55
S_PAB, S_PS0, S_PS1, S_PS2, S_PL0, S_PL1 = 57, 58, 59, 60, 61, 62
S_C = 64          # even: P + i*Q for i = 4..25; odd, from S_C - 1 on: S[]
S_S2, S_C_S0, S_C_S1 = S_C - 1, S_C + 22, S_C + 23   # where the prefix's S[2], S[0], S[1] go
S_INIT3 = 88      # P + 3*Q
S_DUMMY = 89
SLOT_NAME = {S_ACC: 'SLOT_ACC', S_PLO: 'SLOT_PLAIN_LO', S_PHI: 'SLOT_PLAIN_HI', S_EA: 'SLOT_EA',
             S_PAB: 'SLOT_PAB', S_PS0: 'SLOT_PS(0)', S_PS1: 'SLOT_PS(1)', S_PS2: 'SLOT_PS(2)',
             S_PL0: 'SLOT_PL(0)', S_PL1: 'SLOT_PL(1)', S_INIT3: 'SLOT_LM_INIT3',
             S_DUMMY: 'SLOT_LM_DUMMY', S_C: 'SLOT_LM_C(4)', S_S2: 'SLOT_LM_S(-1)',
             S_C + 1: 'SLOT_LM_S(1)', S_C_S0: 'SLOT_LM_S(22)', S_C_S1: 'SLOT_LM_S(23)'}

KEEP, NEXT, CLEAR = 0, 1, 2
AM_NAME = ['AM_KEEP', 'AM_NEXT', 'AM_CLEAR']

# Macros (sequences) and instruction templates
MS, MD, ME = 0, 1, 2
MACRO_NAME = ['LM_MACRO_S', 'LM_MACRO_D', 'LM_MACRO_E']
# template index -> (kind, fixed VC)
TEMPLATES = [('addm', A), ('negm', A), ('addm', Y)]
# macro -> (template, simple delay, store delay or None)
MACROS = {MS: (0, 6, 6), MD: (1, 3, None), ME: (2, 5, None)}


def slot_c(n, odd=False):
    return SLOT_NAME.get(n, 'SLOT(%d)' % n) + (' + LM_ODD' if odd else '')


class I:
    """One issued instruction: op, operands, and its C text."""
    def __init__(self, op, text, **kw):
        self.op, self.text = op, text
        self.__dict__.update(kw)


def r(n):
    return REG_NAME[n]


def iadd(c, d):      # d += c
    return I('add', 'SFPIADD(0, %s, %s, IADD_REG)' % (r(c), r(d)), c=c, d=d)


def iadd_base(d):    # d += base (run time)
    return I('addbase', None, d=d)


def shl_imm(n, c, d):  # d = c << n (n < 0: >> -n)
    return I('shimm', 'SFPSHFT(%d, %s, %s, SHFT_IMM_VC)' % (n, r(c), r(d)), n=n, c=c, d=d)


def shft(c, d):      # d = d << c (c < 0: >> -c), mod 32
    return I('shft', 'SFPSHFT(0, %s, %s, SHFT_REG)' % (r(c), r(d)), c=c, d=d)


def shft2(b, c, d):  # d = b << c
    return I('shft2', 'SFPSHFT2(%s, %s, %s, SHFT2_VB)' % (r(b), r(c), r(d)), b=b, c=c, d=d)


def bor(c, d):       # d |= c
    return I('or', 'SFPOR(0, %s, %s, LOGIC_VD)' % (r(c), r(d)), c=c, d=d)


def bxor(c, d):      # d ^= c
    return I('xor', 'SFPXOR(%s, %s)' % (r(c), r(d)), c=c, d=d)


def neg(c, d):       # d = c ^ 2^31
    return I('neg', 'SFPMOVNEG(%s, %s)' % (r(c), r(d)), c=c, d=d)


def lz(c, d):
    return I('lz', 'SFPLZ(%s, %s)' % (r(c), r(d)), c=c, d=d)


def load(d, slot, am=KEEP, odd=False):
    return I('load', 'SFPLOAD_AM(%s, %s, %s)' % (r(d), AM_NAME[am], slot_c(slot, odd)),
             d=d, slot=slot, am=am, odd=odd)


def store(d, slot, am=KEEP, verify=False):   # to the odd columns
    return I('store', 'SFPSTORE_AM(%s, %s, %s)' % (r(d), AM_NAME[am], slot_c(slot)),
             d=d, slot=slot, am=am, verify=verify)


def lm(macro, d, slot, am=KEEP, odd=False):
    return I('lm', 'SFPLOADMACRO(%s, %s, %s, %s)' % (MACRO_NAME[macro], r(d), AM_NAME[am], slot_c(slot, odd)),
             macro=macro, d=d, slot=slot, am=am, odd=odd)


UNIT = {'add': 'simple', 'addbase': 'simple', 'shimm': 'simple', 'shft': 'simple', 'or': 'simple',
        'xor': 'simple', 'neg': 'simple', 'lz': 'simple', 'shft2': 'round', 'load': 'load',
        'lm': 'load', 'store': 'store'}


# --- The batch ------------------------------------------------------------

def xcur(g):
    return X0 if g % 2 == 0 else X1


def xnext(g):
    return X1 if g % 2 == 0 else X0


def step(g, lm_slot, lm_am=NEXT, odd=True):
    """Key schedule step g (global index: first pass -23..-1, second 0..25,
    third 26..50), updating L[(g + 2) % 3]. Entered with xcur = S[i] + A
    (the add scheduled by the step before) and xnext = S[i + 1] loaded."""
    xc = xcur(g)
    lj, lp = LREG[(g + 2) % 3], LREG[(g + 1) % 3]
    # The scheduled xnext += A and store of A run with the dummy load, whose
    # scheduled G = -A runs with the next load
    return [iadd(lp, xc), shl_imm(3, xc, Y), shl_imm(-29, xc, A), bor(Y, A),
            lm(MD, G, S_DUMMY, KEEP), iadd(lp, A), iadd(A, lj), shft2(lj, A, Y),
            lm(MS, xc, lm_slot, lm_am, odd), shft(G, lj), bor(Y, lj)]


def half(k, am=NEXT, last=False):
    """Encryption half round k = 2..24: d = rotl(d ^ o, o) + S[k]. The three
    registers holding d, o and the loaded S[k] rotate: the sum lands in the
    register S[k] was loaded into."""
    regs = [X0, X1, G]
    d, o = regs[(k + 1) % 3], regs[(k + 2) % 3]
    out = [bxor(o, d), shft2(d, o, Y), neg(o, L0), shft(L0, d), bor(d, Y)]
    if not last:
        out.append(lm(ME, d, S_C + 1, am, odd=True))
    return out


def build():
    """Returns the batch as a list of segments: ('flat', [I]) or
    ('loop', count, [I])."""
    seg = []
    # Prologue: L[2] = rotl(hi + ab, ab) with hi = base + lane, the prefix,
    # and the prefix's S[0..2] to where the passes load them from. The
    # SFPLOADMACRO must come six instructions before the first step's dummy
    # load, and nothing after it may use a fixed Dst address.
    o = True
    pro = [shl_imm(-1, LANE2, L2), iadd_base(L2), load(G, S_PAB, odd=o), iadd(G, L2), shft2(L2, G, Y),
           neg(G, G), shft(G, L2),
           load(A, S_PS2, odd=o), load(L0, S_PL0, odd=o), load(L1, S_PL1, odd=o),
           load(X0, S_PS1, odd=o), store(X0, S_C_S1), load(X0, S_PS0, odd=o), store(X0, S_C_S0),
           store(A, S_S2), load(xcur(-23), S_INIT3),
           lm(MS, xnext(-23), S_C, NEXT), iadd(A, xcur(-23)), bor(Y, L2)]
    seg.append(('flat', 'prologue', pro))
    # First pass: steps 3..25 are g = -23..-1. They load P + i*Q from the
    # even columns, then S[0] and S[1] from the odd columns after them.
    body = sum((step(g, S_C, odd=False) for g in range(-23, -17)), [])
    seg.append(('loop', 3, 'first pass, steps 3..20', body))
    tail = sum((step(g, S_C, odd=False) for g in range(-5, -2)), [])
    tail += step(-2, S_C) + step(-1, S_C, CLEAR)
    seg.append(('flat', 'first pass, steps 21..25', tail))
    # Second pass, steps 0..23
    body = sum((step(g, S_S2) for g in range(0, 6)), [])
    seg.append(('loop', 4, 'second pass, steps 0..23', body))
    # steps 24 and 25 load S[0] and S[1] for the third pass
    seg.append(('flat', 'second pass, steps 24 and 25', step(24, S_S2, CLEAR) + step(25, S_S2, KEEP)))
    # Third pass, steps 0..23
    body = sum((step(g, S_C) for g in range(26, 32)), [])
    seg.append(('loop', 4, 'third pass, steps 0..23', body))
    # Step 24: S[24] only. Then eA = plain_lo + S[0], eB = plain_hi + S[1];
    # the first load is where step 23's scheduled instructions run.
    g = 50
    xc, lp = xcur(g), LREG[(g + 1) % 3]
    regs = [X0, X1, G]
    ea, eb = regs[(2 + 1) % 3], regs[(2 + 2) % 3]     # as half(2) expects them
    w2 = [x for x in regs if x not in (ea, eb)][0]
    assert ea == xc and eb == xnext(g)   # xc is free after its last use; xnext after the scheduled add
    fin = [iadd(lp, xc), shl_imm(3, xc, Y), shl_imm(-29, xc, A), bor(Y, A),
           load(ea, S_PLO - 24, CLEAR),          # RWC is 24 here
           load(Y, S_S2, odd=o), iadd(Y, ea),
           load(eb, S_PHI), load(Y, S_C, odd=o), iadd(Y, eb),
           lm(ME, w2, S_C + 1, NEXT, odd=o)]
    seg.append(('flat', 'third pass, step 24, and the start of the encryption', fin))
    # The registers of a half round repeat every three half rounds, so three
    # of them fit the replay buffer. Replaying them leaves TRISC1 well ahead
    # of the SFPU at the end of a batch, with time for its own work.
    body = sum((half(k) for k in range(2, 5)), [])
    assert [i.text for i in body] == [i.text for i in sum((half(k) for k in range(5, 8)), [])]
    seg.append(('replay', 7, 'half rounds with S[2..22]', body))
    tail = half(23, am=CLEAR)
    tail += half(24, last=True)
    seg.append(('flat', 'half rounds with S[23] and S[24]', tail))
    # The last scheduled add runs with the load of ACC. Then
    # ACC = (ACC << 1) | (eA == cypher_lo), via clz(eA ^ cypher_lo) >> 5
    regs3 = [X0, X1, G]
    d24 = regs3[(24 + 1) % 3]
    o24 = regs3[(24 + 2) % 3]
    res = [x for x in regs3 if x not in (d24, o24)][0]
    cmpseq = [load(L1, S_ACC, odd=True), store(res, S_EA, verify=True), bxor(CLO, res), lz(res, res),
              shl_imm(-5, res, res), shl_imm(1, L1, L1), bor(res, L1), store(L1, S_ACC)]
    seg.append(('flat', 'compare', cmpseq))
    return seg, res


def flatten(seg):
    out = []
    for s in seg:
        if s[0] == 'flat':
            out += s[2]
        else:
            out += s[3] * s[1]
    return out


def replay_body(seg):
    bodies = [s[3] for s in seg if s[0] == 'replay']
    assert len(bodies) == 1 and len(bodies[0]) <= 32
    return bodies[0]


# --- Model -------------------------------------------------------------------

def rotl(x, s):
    s &= 31
    return ((x << s) | (x >> (32 - s))) & M32 if s else x


def sshift(x, amount):
    """SFPSHFT/SFPSHFT2: left for amount >= 0, logical right otherwise"""
    if amount & 0x80000000:
        return x >> ((-amount) & 31)
    return (x << (amount & 31)) & M32


class Model:
    def __init__(self, lanes=4):
        self.n = lanes
        self.reg = {i: [0] * lanes for i in range(8)}
        self.reg[LANE2] = [2 * i for i in range(lanes)]
        self.reg[CLO] = [0] * lanes
        self.dst = {}
        self.rwc = 0
        self.sched = []     # (fire_at, unit, kind, operands)
        self.t = 0
        self.errors = []

    def rd(self, x):
        return self.reg[x]

    def effects(self, ins, base=0, verify=False):
        """Returns (reads, writes{reg: values}, dst_write or None, rwc_op)"""
        op = ins.op
        g = self.rd
        if op == 'add':
            return {ins.c, ins.d}, {ins.d: [(a + b) & M32 for a, b in zip(g(ins.c), g(ins.d))]}, None
        if op == 'addbase':
            return {ins.d}, {ins.d: [(a + base) & M32 for a in g(ins.d)]}, None
        if op == 'shimm':
            return {ins.c}, {ins.d: [sshift(a, ins.n & M32) for a in g(ins.c)]}, None
        if op == 'shft':
            return {ins.c, ins.d}, {ins.d: [sshift(a, b) for a, b in zip(g(ins.d), g(ins.c))]}, None
        if op == 'shft2':
            return {ins.b, ins.c}, {ins.d: [sshift(a, b) for a, b in zip(g(ins.b), g(ins.c))]}, None
        if op == 'or':
            return {ins.c, ins.d}, {ins.d: [a | b for a, b in zip(g(ins.c), g(ins.d))]}, None
        if op == 'xor':
            return {ins.c, ins.d}, {ins.d: [a ^ b for a, b in zip(g(ins.c), g(ins.d))]}, None
        if op == 'neg':
            return {ins.c}, {ins.d: [a ^ 0x80000000 for a in g(ins.c)]}, None
        if op == 'lz':
            return {ins.c}, {ins.d: [32 - a.bit_length() for a in g(ins.c)]}, None
        if op in ('load', 'lm'):
            addr = (ins.slot + self.rwc, ins.odd)
            if addr not in self.dst:
                self.errors.append('t=%d: load of unwritten slot %s' % (self.t, addr))
                vals = [0xDEAD0000 + self.t] * self.n
            else:
                vals = list(self.dst[addr])
            return set(), {ins.d: vals}, None
        if op == 'store':
            if ins.verify and not verify:
                return set(), {}, None
            return {ins.d}, {}, ((ins.slot + self.rwc, True), list(g(ins.d)))
        raise ValueError(op)

    def issue(self, ins, base=0, verify=False):
        due = [s for s in self.sched if s[0] == self.t]
        self.sched = [s for s in self.sched if s[0] != self.t]
        units = [UNIT[ins.op]]
        all_reads, all_writes, dst_writes = [], [], []
        rd, wr, dw = self.effects(ins, base, verify)
        all_reads.append(rd)
        all_writes.append(wr)
        if dw:
            dst_writes.append(dw)
        for (_, unit, kind, ops) in due:
            if unit in units:
                self.errors.append('t=%d: two instructions on the %s sub-unit' % (self.t, unit))
            units.append(unit)
            if kind == 'addm':     # m = vc + m
                vc, m = ops
                all_reads.append({vc, m})
                all_writes.append({m: [(a + b) & M32 for a, b in zip(self.rd(vc), self.rd(m))]})
            elif kind == 'negm':   # m = vc ^ 2^31
                vc, m = ops
                all_reads.append({vc})
                all_writes.append({m: [a ^ 0x80000000 for a in self.rd(vc)]})
            elif kind == 'store':  # store reg to addr
                reg, addr = ops
                all_reads.append({reg})
                all_writes.append({})
                dst_writes.append((addr, list(self.rd(reg))))
        if 'simple' in units and 'round' in units:
            self.errors.append('t=%d: simple and round sub-units in one cycle' % self.t)
        # instructions sharing a cycle must not depend on each other
        for i in range(len(all_writes)):
            for j in range(len(all_writes)):
                if i != j and (set(all_writes[i]) & (all_reads[j] | set(all_writes[j]))):
                    self.errors.append('t=%d: dependent instructions share a cycle (%s)' %
                                       (self.t, ins.text))
        for w in all_writes:
            for k, v in w.items():
                self.reg[k] = v
        for addr, v in dst_writes:
            self.dst[addr] = v
        if ins.op == 'lm':
            tmpl, sdelay, stdelay = MACROS[ins.macro]
            kind, vc = TEMPLATES[tmpl]
            addr = (ins.slot + self.rwc, True)   # the scheduled store's column
            new = [(self.t + 1 + sdelay, 'simple', kind, (vc, ins.d))]
            if stdelay is not None:
                new.append((self.t + 1 + stdelay, 'store', 'store', (A, addr)))
            for n in new:
                if any(s[0] == n[0] and s[1] == n[1] for s in self.sched):
                    self.errors.append('t=%d: scheduled instruction replaced' % self.t)
                self.sched.append(n)
        if ins.op in ('load', 'lm', 'store'):
            if ins.am == NEXT:
                self.rwc += 1
            elif ins.am == CLEAR:
                self.rwc = 0
        self.t += 1


def rc5_lo(hi, mid, lo, plain_lo, plain_hi):
    S = [(P + i * Q) & M32 for i in range(26)]
    L = [lo, mid, hi]
    a = b = 0
    for k in range(78):
        a = S[k % 26] = rotl((S[k % 26] + a + b) & M32, 3)
        b = L[k % 3] = rotl((L[k % 3] + a + b) & M32, (a + b) & M32)
    ea = (plain_lo + S[0]) & M32
    eb = (plain_hi + S[1]) & M32
    for i in range(1, 13):
        ea = (rotl(ea ^ eb, eb) + S[2 * i]) & M32
        eb = (rotl(eb ^ ea, ea) + S[2 * i + 1]) & M32
    return ea


def prefix(mid, lo):
    a = rotl(P, 3)
    b = rotl((lo + a) & M32, a)
    s0, l0 = a, b
    a = rotl((P + Q + a + b) & M32, 3)
    b = rotl((mid + a + b) & M32, (a + b) & M32)
    s1, l1 = a, b
    a = rotl((P + 2 * Q + a + b) & M32, 3)
    return s0, s1, a, l0, l1, (a + b) & M32


def check(seg, res, batches):
    stream = flatten(seg)
    rnd = random.Random(72)
    lanes = 8
    m = Model(lanes)
    ev, od = False, True
    for i in range(4, 26):
        m.dst[(S_C + i - 4, ev)] = [(P + i * Q) & M32] * lanes
    m.dst[(S_INIT3, ev)] = [(P + 3 * Q) & M32] * lanes
    for i in range(26):          # the dummy loads move with the RWC; their values are unused
        m.dst[(S_DUMMY + i, ev)] = [0x5A5A0000 + i] * lanes
    m.dst[(S_ACC, od)] = [0] * lanes
    bad = 0
    for b in range(batches):
        mid, lo, plo, phi = (rnd.getrandbits(32) for _ in range(4))
        base = rnd.randrange(8) * 32
        clo = rc5_lo(base + rnd.randrange(lanes), mid, lo, plo, phi)   # one lane matches
        s0, s1, s2, l0, l1, ab = prefix(mid, lo)
        for slot, v in ((S_PAB, ab), (S_PS0, s0), (S_PS1, s1), (S_PS2, s2), (S_PL0, l0), (S_PL1, l1)):
            m.dst[(slot, od)] = [v] * lanes
        m.dst[(S_PLO, ev)] = [plo] * lanes
        m.dst[(S_PHI, ev)] = [phi] * lanes
        m.reg[CLO] = [clo] * lanes
        acc_before = list(m.dst[(S_ACC, od)])
        for ins in stream:
            m.issue(ins, base, verify=True)
        want = [rc5_lo(base + i, mid, lo, plo, phi) for i in range(lanes)]
        if m.dst[(S_EA, od)] != want:
            bad += 1
        acc = [((a << 1) | (w == clo)) & M32 for a, w in zip(acc_before, want)]
        if m.dst[(S_ACC, od)] != acc:
            bad += 1
        if m.rwc != 0 or m.sched:
            m.errors.append('batch %d: RWC %d, %d instructions still scheduled' % (b, m.rwc, len(m.sched)))
        # the constants must survive
        if any(m.dst[(S_C + i - 4, ev)] != [(P + i * Q) & M32] * lanes for i in range(4, 26)):
            m.errors.append('batch %d: constants overwritten' % b)
    return len(stream), bad, m.errors


# --- Output --------------------------------------------------------------------

def emit(seg, res, nstream):
    o = []
    w = o.append
    w('''/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Generated by tools/gen_sfpu_lm.py, which also checks this instruction
 * stream against a model of SFPLOADMACRO and a plain RC5-72. Do not edit.
 *
 * RC5-72 on the Tensix Vector Unit (SFPU) with SFPLOADMACRO: as
 * sfpu_batch() in r72tt_sfpu.h, in %d instead of about 1100 cycles per 32
 * keys. Not for the ttsim simulator, which does not implement SFPLOADMACRO.
 *
 * SFPLOADMACRO loads a register and schedules up to one instruction per
 * sub-unit of the SFPU, which then run in the same cycle as an instruction
 * issued a set number of instructions later. A scheduled instruction can
 * only write the register its SFPLOADMACRO loaded ("m"), and a scheduled
 * store goes to the address that was loaded from. Here:
 * - S[i + 2] is loaded during step i, into the register that held
 *   S[i] + A: two registers take turns. That SFPLOADMACRO schedules
 *   m += A (A = S[i + 1] by then) and the store of A to the address it
 *   loaded from, so each pass leaves its S[] one slot further on.
 * - Both run in the cycle of a second, dummy SFPLOADMACRO, which schedules
 *   the sign flip of the rotate amount to run in the cycle of the next load.
 * - The first pass loads P + i*Q instead. Those constants are in the even
 *   columns of their Dst slots, and with DEST_WR_COL_EXCHANGE set every
 *   store goes to the odd columns, so the scheduled stores leave them
 *   alone. Everything stored is therefore loaded from the odd columns
 *   (address + LM_ODD), and TRISC1 reads it back from there.
 * - The third pass stores S[] like the second, and the encryption loads it:
 *   each half round's load schedules the add of S[i], which runs in the
 *   cycle of the next half round's load.
 * Delays count issued SFPU instructions, so the stream does not depend on
 * how fast it is pushed. Instructions that share a cycle are independent.
 *
 * LRegs: 0 A (and the scheduled stores' source), 1 and 2 S[i] + A in turn,
 * 3 the sign-flipped rotate amount, 4 a rotate's left half, 5..7 L[0..2].
 * The encryption has eA, eB and S[i] in LRegs 1..3 in rotation.
 *
 * Dst slots: 64..85 hold P + i*Q for i = 4..25 in their even columns, and
 * 63..87 hold S[] in their odd columns: the first pass leaves S[i] in slot
 * 60 + i, and the encryption finds it in slot 63 + i. 88 P + 3*Q, 89 on the
 * dummy loads (their address moves with the Dst RWC); the others as in
 * r72tt_sfpu.h.
*/

#ifndef R72TT_SFPU_LM_H
#define R72TT_SFPU_LM_H

#define LM_A   0
#define LM_X0  1
#define LM_X1  2
#define LM_G   3
#define LM_Y   4
#define LM_L0  5
#define LM_L1  6
#define LM_L2  7

#define SLOT_LM_C(i)   SLOT(%d + (i) - 4)   /* P + i*Q, i = 4..25: even columns */
#define SLOT_LM_S(n)   SLOT(%d + (n))       /* S[]: odd columns */
#define SLOT_LM_INIT3  SLOT(%d)
#define SLOT_LM_DUMMY  SLOT(%d)
#define LM_ODD         2   /* added to a load's address: the odd columns */

/* SFPLOADMACRO: load LReg d from Dst and schedule the instructions of
   sequence 'macro'. The low address bit is the top bit of d instead. */
#define SFPLOADMACRO(macro, d, am, addr) \\
  TT_OP(0x93, ((((macro) << 2) + ((d) & 3)) << 20) + (SFP_INT32 << 16) + ((am) << 13) + \\
              (addr) + ((d) >> 2))
#define SFPCONFIG_IMM(imm16, d) TT_OP(0x91, ((uint32_t)(imm16) << 8) + ((d) << 4) + 1)

/* Sequences: per sub-unit a byte of 0x80 (the scheduled instruction's other
   operand comes from its template; a store takes LReg 0) + delay * 8 +
   (4 + template, or 3 for a store); the simple sub-unit's byte is the
   lowest and the store's the highest. */
#define LM_MACRO_S   0   /* m += LReg 0 after 6; store LReg 0 after 6 */
#define LM_MACRO_D   1   /* m = LReg 0 ^ 2^31 after 3 */
#define LM_MACRO_E   2   /* m += LReg 4 after 5 */
#define LM_SEQ_S     0xB30000B4u
#define LM_SEQ_D     0x0000009Du
#define LM_SEQ_E     0x000000AEu
/* delays of all sub-units count issued instructions; stores are INT32 */
#define LM_MISC      0xF04u
''' % (nstream, S_C, S_C, S_INIT3, S_DUMMY))
    rb = replay_body(seg)
    w('/* Replay buffer contents: three encryption half rounds */')
    w('#define LM_REPLAY_LEN %d' % len(rb))
    w('static const uint32_t sfpu_lm_replay[LM_REPLAY_LEN] =')
    w('{')
    for ins in rb:
        w('  %s,' % ins.text)
    w('};')
    w('')
    w('''/* One-time setup after sfpu_setup(): templates, sequences, constants, the
   replay buffer (replacing sfpu_setup()'s), and stores to the odd columns
   from here on. An instruction with VD 12 + i is not executed but becomes
   template i. */
static void sfpu_lm_setup(void)
{
  uint32_t i;

  TTI(SFPIADD(0, LM_A, 12, IADD_REG));
  TTI(SFPMOVNEG(LM_A, 13));
  TTI(SFPIADD(0, LM_Y, 14, IADD_REG));
  TTI(SFPNOP);
  sfpu_loadi32(0, LM_SEQ_S);
  TTI(SFPCONFIG(4 + LM_MACRO_S));
  TTI(SFPNOP);
  sfpu_loadi32(0, LM_SEQ_D);
  TTI(SFPCONFIG(4 + LM_MACRO_D));
  TTI(SFPNOP);
  sfpu_loadi32(0, LM_SEQ_E);
  TTI(SFPCONFIG(4 + LM_MACRO_E));
  TTI(SFPNOP);
  TTI(SFPCONFIG_IMM(LM_MISC, 8));
  TTI(SFPNOP);

  sfpu_loadi32(LM_X0, 0xB7E15163u + 3 * 0x9E3779B9u);
  TTI(SFPSTORE(LM_X0, SLOT_LM_INIT3));
  for (i = 4; i < 26; i++)
  {
    sfpu_loadi32(LM_X0, 0xB7E15163u + i * 0x9E3779B9u);
    tt_push(SFPSTORE(LM_X0, SLOT_LM_C(i)));
  }
  TTI(SFPCONFIG_LANE(LANECONFIG_DEST_WR_COL_EXCHANGE));
  TTI(SFPNOP);
  sfpu_clear_acc();

  /* Recording with execution off: the recorded instructions must be pushed
     one store each, never as fused .ttinsn sequences */
  tt_push(REPLAY(0, LM_REPLAY_LEN, 0, 1));
  for (i = 0; i < LM_REPLAY_LEN; i++)
    tt_push(sfpu_lm_replay[i]);
}

/* Queues one batch; see sfpu_batch() */
static void sfpu_batch_lm(uint32_t base, int verify)
{
  uint32_t i;
''')
    for s in seg:
        if s[0] == 'flat':
            w('  /* %s */' % s[1])
            for ins in s[2]:
                emit_ins(w, ins, '  ')
        elif s[0] == 'replay':
            w('  /* %s */' % s[2])
            w('#pragma GCC unroll 1')
            w('  for (i = 0; i < %d; i++)' % s[1])
            w('    TTI(REPLAY(0, LM_REPLAY_LEN, 0, 0));')
        else:
            w('  /* %s */' % s[2])
            w('#pragma GCC unroll 1')
            w('  for (i = 0; i < %d; i++)' % s[1])
            w('  {')
            for ins in s[3]:
                emit_ins(w, ins, '    ')
            w('  }')
        w('')
    o.pop()
    w('}')
    w('')
    w('#endif /* R72TT_SFPU_LM_H */')
    return '\n'.join(o) + '\n'


def emit_ins(w, ins, ind):
    if ins.op == 'addbase':
        w('%stt_push(SFPIADD(base, %s, %s, IADD_IMM));' % (ind, r(ins.d), r(ins.d)))
    elif ins.op == 'store' and ins.verify:
        w('%sif (verify)' % ind)
        w('%s  TTI(%s);' % (ind, ins.text))
    else:
        w('%sTTI(%s);' % (ind, ins.text))


def main():
    batches = 300
    if len(sys.argv) > 2 and sys.argv[1] == '-n':
        batches = int(sys.argv[2])
    seg, res = build()
    n, bad, errors = check(seg, res, batches)
    for e in errors[:20]:
        sys.stderr.write(e + '\n')
    sys.stderr.write('%d instructions per batch; %d batches of 8 lanes checked, %d wrong, %d rule violations\n'
                     % (n, batches, bad, len(errors)))
    if bad or errors:
        sys.exit(1)
    sys.stdout.write(emit(seg, res, n))


if __name__ == '__main__':
    main()
