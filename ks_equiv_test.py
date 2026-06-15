"""Prove the Kogge-Stone slider attack map is IDENTICAL to pgx's gather-based slider detection
(_is_attacked's attacked_far) on every square of thousands of random boards, BEFORE touching chess.py."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import pgx._src.games.chess as C

# ---- pgx's slider attack detection for one square (replicated verbatim from _is_attacked.attacked_far)
def pgx_slider_attacked(board, occ, pos):
    def attacked_far(to):
        ok = (to >= 0) & (board[to] < 0)
        piece = jnp.abs(board[to])
        ok &= (piece == C.QUEEN) | (piece == C.ROOK) | (piece == C.BISHOP)
        ok &= C.CAN_MOVE[piece, pos, to] & ~jnp.any(occ & C.BETWEEN_MASK[pos, to] != 0)
        return ok
    return jax.vmap(attacked_far)(C.LEGAL_DEST_FAR[pos, :]).any()

# ---- the Kogge-Stone slider attack map (board (64,) with sq = file*8 + rank -> reshape(8,8)=[file,rank])
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

def shift(x, dr, dc):
    x = jnp.roll(x, (dr, dc), axis=(-2, -1))
    if dr > 0:   x = x.at[..., :dr, :].set(False)
    elif dr < 0: x = x.at[..., dr:, :].set(False)
    if dc > 0:   x = x.at[..., :, :dc].set(False)
    elif dc < 0: x = x.at[..., :, dc:].set(False)
    return x

def fill_dir(g, e, dr, dc):
    g = g | (e & shift(g, dr, dc))
    e = e & shift(e, dr, dc)
    g = g | (e & shift(g, 2 * dr, 2 * dc))
    e = e & shift(e, 2 * dr, 2 * dc)
    g = g | (e & shift(g, 4 * dr, 4 * dc))
    return shift(g, dr, dc)

def ks_slider_map(board):
    g = board.reshape(8, 8)
    occ = g != C.EMPTY
    e = ~occ
    ortho = (g == -C.ROOK) | (g == -C.QUEEN)   # opponent (negative) rooks/queens
    diag = (g == -C.BISHOP) | (g == -C.QUEEN)  # opponent bishops/queens
    a = jnp.zeros_like(occ)
    for dr, dc in ORTHO:
        a = a | fill_dir(ortho, e, dr, dc)
    for dr, dc in DIAG:
        a = a | fill_dir(diag, e, dr, dc)
    return a.reshape(64)

def random_slider_board(rng):
    # only sliders (±B, ±R, ±Q) + empty -> then full _is_attacked == pure slider attack detection,
    # so the Kogge-Stone map (which covers near AND far slider attacks) can be compared head-on.
    b = np.zeros(64, np.int8)
    occ = rng.random(64) < 0.30
    vals = rng.choice([-C.QUEEN, -C.ROOK, -C.BISHOP, C.BISHOP, C.ROOK, C.QUEEN], 64)
    b[occ] = vals[occ]
    return b

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 2000
    ks_fn = jax.jit(ks_slider_map)
    occ_fn = jax.jit(C._occupancy)
    # full pgx _is_attacked over all 64 squares (on slider-only boards = pure slider detection)
    pgx_fn = jax.jit(jax.vmap(C._is_attacked, in_axes=(None, None, 0)))
    mism = 0
    for _ in range(N):
        b = jnp.array(random_slider_board(rng))
        occ = occ_fn(b)
        kmap = ks_fn(b)
        pmap = pgx_fn(b, occ, jnp.arange(64))
        mism += int(jnp.sum(kmap != pmap))
    print(f"slider-map equivalence: {N} slider-only boards x 64 squares, mismatches = {mism}  -> {'IDENTICAL' if mism == 0 else 'DIVERGES'}")
