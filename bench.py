"""
Benchmark: steps/sec for each game at representative batch sizes.
Run with:  .venv-metal/bin/python bench.py [tag]

Uses CPU backend for reproducible, non-hanging timing.
Metal JIT compilation times are unreliable for benchmarking; CPU numbers
show the algorithmic improvements cleanly.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time, sys, json, pathlib
import jax
import jax.numpy as jnp

import pgx

GAMES = {
    "chess":      ("chess",      [256, 1024]),
    "go_9x9":     ("go_9x9",     [512, 2048]),
    "go_19x19":   ("go_19x19",   [64,  256]),
    "shogi":      ("shogi",      [128, 512]),
    "backgammon": ("backgammon", [512, 2048]),
}

WARMUP  = 3
MEASURE = 10


def bench_game(name: str, batch: int) -> float:
    env = pgx.make(name)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch)
    state = jax.jit(jax.vmap(env.init))(keys)

    step_vmap = jax.jit(jax.vmap(env.step))

    @jax.jit
    def step_batch(state, key):
        key, subkey = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        subkeys = jax.random.split(subkey, batch)
        actions = jax.vmap(lambda l, k: jax.random.categorical(k, l))(logits, subkeys)
        return step_vmap(state, actions, subkeys), key

    rng = jax.random.PRNGKey(0)
    for _ in range(WARMUP):
        state, rng = step_batch(state, rng)
    jax.block_until_ready(state.rewards)

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        state, rng = step_batch(state, rng)
    jax.block_until_ready(state.rewards)
    elapsed = time.perf_counter() - t0

    return batch * MEASURE / elapsed


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print(f"\n{'='*60}")
    print(f"  pgx bench [{tag}]  JAX {jax.__version__}  {jax.devices()[0]}")
    print(f"{'='*60}")
    print(f"{'game':<16} {'batch':>6}  {'steps/sec':>12}")
    print(f"{'-'*40}")

    results = {}
    for game, (pgx_name, batches) in GAMES.items():
        for batch in batches:
            try:
                sps = bench_game(pgx_name, batch)
                label = f"{game}@{batch}"
                results[label] = sps
                print(f"{game:<16} {batch:>6}  {sps:>12,.0f}")
                sys.stdout.flush()
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"{game:<16} {batch:>6}  ERROR: {e}")

    out = pathlib.Path(f"bench_{tag}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
