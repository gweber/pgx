"""Kogge-Stone (occluded-fill) sliding-piece attack map vs the gather/ray baseline that pgx's chess
movegen uses today. Branch-free, table-free fills expressed as (8,8) grid shifts — the GPU/SIMD-ideal
form. Verifies both agree, then benchmarks throughput. CPU backend (matches bench.py convention)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time, sys
import numpy as np
import jax, jax.numpy as jnp

ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRS = ORTHO + DIAG

# ---------------------------------------------------------------- Kogge-Stone (the proposed kernel)
def shift(x, dr, dc):
    """grid shift by (dr,dc), off-board filled False (no file-wrap — the (8,8) form's whole point)."""
    x = jnp.roll(x, (dr, dc), axis=(-2, -1))
    if dr > 0:   x = x.at[..., :dr, :].set(False)
    elif dr < 0: x = x.at[..., dr:, :].set(False)
    if dc > 0:   x = x.at[..., :, :dc].set(False)
    elif dc < 0: x = x.at[..., :, dc:].set(False)
    return x

def fill_dir(g, e, dr, dc):
    """occluded fill: propagate sliders g through empties e, log-step doubling; return attacked set."""
    g = g | (e & shift(g, dr, dc))
    e = e & shift(e, dr, dc)
    g = g | (e & shift(g, 2 * dr, 2 * dc))
    e = e & shift(e, 2 * dr, 2 * dc)
    g = g | (e & shift(g, 4 * dr, 4 * dc))
    return shift(g, dr, dc)  # one more step -> attacked squares (incl. blocker, excl. slider)

def kogge_attacks(occ, ortho, diag):
    e = ~occ
    a = jnp.zeros_like(occ)
    for dr, dc in ORTHO:
        a = a | fill_dir(ortho, e, dr, dc)
    for dr, dc in DIAG:
        a = a | fill_dir(diag, e, dr, dc)
    return a

# ---------------------------------------------------------------- gather/ray baseline (pgx-style)
RAY = np.zeros((64, 8, 7), np.int32)
RAYVALID = np.zeros((64, 8, 7), bool)
for sq in range(64):
    r, c = divmod(sq, 8)
    for di, (dr, dc) in enumerate(DIRS):
        rr, cc, k = r + dr, c + dc, 0
        while 0 <= rr < 8 and 0 <= cc < 8:
            RAY[sq, di, k] = rr * 8 + cc; RAYVALID[sq, di, k] = True
            k += 1; rr += dr; cc += dc
RAY_j = jnp.array(RAY)
RAYVALID_j = jnp.array(RAYVALID)
DIR_ORTHO = jnp.array([True] * 4 + [False] * 4)

def gather_attacks(occ_f, ortho_f, diag_f):
    """occ_f/ortho_f/diag_f: (...,64) bool. Gather ray occupancy, cumsum to first blocker, scatter."""
    occ_ray = occ_f[..., RAY_j]                                  # (...,64,8,7) gather (the hot op)
    before = jnp.cumsum(occ_ray.astype(jnp.int32), -1) - occ_ray.astype(jnp.int32)
    attacked = RAYVALID_j & (before == 0)                        # squares reachable along each ray
    slider = jnp.where(DIR_ORTHO[:, None], ortho_f[..., None, None], diag_f[..., None, None])  # gate
    attacked = attacked & jnp.broadcast_to(slider, attacked.shape)
    flat_idx = RAY_j.reshape(-1)
    amap = jnp.zeros(occ_f.shape, bool)
    amap = amap.at[..., flat_idx].max(attacked.reshape(*attacked.shape[:-3], -1))
    return amap

# ---------------------------------------------------------------- random boards + verify + benchmark
def make_boards(B, seed=0):
    rng = np.random.default_rng(seed)
    occ = rng.random((B, 8, 8)) < 0.30
    sl = occ & (rng.random((B, 8, 8)) < 0.5)      # half the occupied squares are sliders
    isdiag = rng.random((B, 8, 8)) < 0.5
    ortho = sl & ~isdiag
    diag = sl & isdiag
    return jnp.array(occ), jnp.array(ortho), jnp.array(diag)

def bench(fn, args, warmup=3, iters=20):
    f = jax.jit(fn)
    for _ in range(warmup): jax.block_until_ready(f(*args))
    t = time.perf_counter()
    for _ in range(iters): r = f(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t) / iters

if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    occ, ortho, diag = make_boards(B)
    occ_f, ortho_f, diag_f = (x.reshape(B, 64) for x in (occ, ortho, diag))

    a_kog = jax.jit(kogge_attacks)(occ, ortho, diag).reshape(B, 64)
    a_gat = jax.jit(gather_attacks)(occ_f, ortho_f, diag_f)
    agree = bool(jnp.all(a_kog == a_gat))
    print(f"correctness: kogge == gather on {B} random boards -> {agree}")
    if not agree:
        diff = int(jnp.sum(a_kog != a_gat)); print(f"  MISMATCH in {diff} squares"); sys.exit(1)

    t_kog = bench(kogge_attacks, (occ, ortho, diag))
    t_gat = bench(gather_attacks, (occ_f, ortho_f, diag_f))
    print(f"batch={B}  kogge-stone {t_kog*1e3:7.3f} ms   gather/ray {t_gat*1e3:7.3f} ms   speedup {t_gat/t_kog:.2f}x")
    print(f"          {B/t_kog/1e6:8.2f} M boards/s   vs   {B/t_gat/1e6:8.2f} M boards/s")
