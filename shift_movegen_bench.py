"""Shift-based pseudo-legal slider+knight move-PLANE generation vs pgx's gather reference.

The 4672 action layout (from*73 + plane) is itself a [from x direction-distance] grid, so move
generation maps naturally to branch-free (8,8) grid shifts -- the SIMT-ideal form -- with NO
CAN_MOVE / BETWEEN_MASK gathers and NO per-(from,to) _to_label gather.

This prototype proves the shift form reproduces the gather form bit-exactly on the gather-heavy
core (queen/rook/bishop sliders with path checks + knight leapers), then benchmarks throughput.
Pins / check-target / king-danger are cheap bitmask AND-overlays added on top later.

sq = file*8 + rank ; board.reshape(8,8) = [file, rank] (axis0=file, axis1=rank).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time, sys
import numpy as np
import jax, jax.numpy as jnp
import pgx._src.games.chess as C
from pgx._src.games.chess import KNIGHT, BISHOP, ROOK, QUEEN, EMPTY

FP = np.asarray(C.FROM_PLANE)  # (64,73)

# ---- derive plane lookups from FROM_PLANE geometry --------------------------------------------
SLIDER = {}   # (uf,ur,k) -> plane
KNIGHTP = {}  # (df,dr) -> plane
for p in range(9, 65):
    for frm in range(64):
        to = FP[frm, p]
        if to >= 0:
            df, dr = to // 8 - frm // 8, to % 8 - frm % 8
            SLIDER[(np.sign(df), np.sign(dr), max(abs(df), abs(dr)))] = p
            break
for p in range(65, 73):
    for frm in range(64):
        to = FP[frm, p]
        if to >= 0:
            KNIGHTP[(to // 8 - frm // 8, to % 8 - frm % 8)] = p
            break

DIRS = sorted({(uf, ur) for (uf, ur, _) in SLIDER})  # 8 slider directions

# ---- branch-free grid shift (no wrap), axis0=file, axis1=rank ---------------------------------
def shift(x, a0, a1):
    x = jnp.roll(x, (a0, a1), axis=(-2, -1))
    if a0 > 0:   x = x.at[..., :a0, :].set(False)
    elif a0 < 0: x = x.at[..., a0:, :].set(False)
    if a1 > 0:   x = x.at[..., :, :a1].set(False)
    elif a1 < 0: x = x.at[..., :, a1:].set(False)
    return x

def back(x, uf, ur, n):
    """bring value of square (from + n*(uf,ur)) to from's position (off-board -> False)."""
    return shift(x, -n * uf, -n * ur)

# ---- THE SHIFT GENERATOR: (B,64,73) bool, only slider+knight planes set -----------------------
def shift_planes(board):  # board (B,8,8) int8
    mine = board > 0
    empty = board == EMPTY
    not_own = board <= 0
    ortho = mine & ((board == ROOK) | (board == QUEEN))
    diag = mine & ((board == BISHOP) | (board == QUEEN))
    B = board.shape[0]
    cols = [jnp.zeros((B, 64), bool) for _ in range(73)]  # one column per plane; assemble in ONE stack
    for (uf, ur) in DIRS:
        src = ortho if (uf == 0 or ur == 0) else diag
        path = jnp.ones_like(empty)  # squares 1..k-1 clear (k=1 -> trivially True)
        for k in range(1, 8):
            plane_mask = src & path & back(not_own, uf, ur, k)   # (B,8,8) from-mask
            cols[SLIDER[(uf, ur, k)]] = plane_mask.reshape(B, 64)
            path = path & back(empty, uf, ur, k)                  # extend clear path
    knight = mine & (board == KNIGHT)
    for (df, dr), p in KNIGHTP.items():
        # knight move from -> from+(df,dr): legal iff dest not own & on-board (path always clear)
        cols[p] = (knight & shift(not_own, -df, -dr)).reshape(B, 64)
    return jnp.stack(cols, axis=-1)  # (B,64,73) in a single concat -> no repeated big-array scatter

# ---- GATHER REFERENCE: exactly pgx's pseudo-legal ok, indexed by plane ------------------------
FP_j = jnp.asarray(FP)
def gather_planes(board_flat):  # board_flat (B,64) int8
    occ = jax.vmap(C._occupancy)(board_flat)  # (B,2) uint32

    def one(board, occ):
        def per_plane(p):
            def per_from(frm):
                to = FP_j[frm, p]
                piece = board[frm]
                ok = (to >= 0) & (piece > 0) & (board[to] <= 0)
                ok &= C.CAN_MOVE[piece, frm, to]
                ok &= ~jnp.any(occ & C.BETWEEN_MASK[frm, to] != 0)
                # restrict to slider/knight pieces (king/pawn handled elsewhere)
                ok &= (piece == ROOK) | (piece == BISHOP) | (piece == QUEEN) | (piece == KNIGHT)
                ok &= p >= 9  # planes 0..8 are pawn underpromotions, not normal moves
                return ok
            return jax.vmap(per_from)(jnp.arange(64))
        return jax.vmap(per_plane)(jnp.arange(73)).T  # (64,73)
    return jax.vmap(one)(board_flat, occ)

# ---- random slider/knight boards --------------------------------------------------------------
def make_boards(B, seed=0):
    rng = np.random.default_rng(seed)
    b = np.zeros((B, 64), np.int8)
    occ = rng.random((B, 64)) < 0.30
    vals = rng.choice([-QUEEN, -ROOK, -BISHOP, -KNIGHT, KNIGHT, BISHOP, ROOK, QUEEN], (B, 64))
    b[occ] = vals[occ]
    return jnp.asarray(b)

def bench(fn, arg, warmup=3, iters=20):
    f = jax.jit(fn)
    for _ in range(warmup): jax.block_until_ready(f(arg))
    t = time.perf_counter()
    for _ in range(iters): r = f(arg)
    jax.block_until_ready(r)
    return (time.perf_counter() - t) / iters

if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    bf = make_boards(B)
    bg = bf.reshape(B, 8, 8)
    a_shift = jax.jit(shift_planes)(bg)
    a_gath = jax.jit(gather_planes)(bf)
    mism = int(jnp.sum(a_shift != a_gath))
    print(f"correctness: shift == gather on {B} boards x 64 x 73 -> "
          f"{'IDENTICAL' if mism == 0 else f'DIVERGES ({mism} entries)'}")
    if mism:
        sys.exit(1)
    t_s = bench(shift_planes, bg)
    t_g = bench(gather_planes, bf)
    print(f"batch={B}  shift {t_s*1e3:8.3f} ms   gather {t_g*1e3:8.3f} ms   speedup {t_g/t_s:.2f}x")
    print(f"          {B/t_s/1e6:8.2f} M boards/s   vs   {B/t_g/1e6:8.2f} M boards/s")
