# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Random-legal self-play sanity check / micro-benchmark for ``layer_go``.

Runs many episodes of uniformly-random *legal* self-play (vmapped + jitted, one
``jax.lax.while_loop`` that stops when every game in the batch has terminated) and
prints first-order health metrics: game length, win rates, score margin, terminal-
reason distribution, branching factor, and a rough env-steps/sec throughput.

This does NOT change any game rules. It only samples from ``legal_action_mask`` and
reads public state, plus the internal ``pgx.layer_go._score`` helper for the area
score (used only for analysis output, not for play).

Usage:
    python examples/layer_go_random_selfplay.py --episodes 256 --seed 0
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import pgx
from pgx.layer_go import KOMI, MAX_GAME_LENGTH, _score

env = pgx.make("layer_go")


def _sample_legal_action(key, legal_action_mask):
    """Uniformly sample one legal action (illegal actions get ``-inf`` logit)."""
    logits = jnp.where(legal_action_mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


@jax.jit
def _run_batch(key, init_keys):
    """Play one batch of random-legal games to termination.

    ``init_keys`` has shape ``(num_episodes, 2)``; its leading size fixes the batch.
    Returns per-episode arrays for host-side aggregation.
    """
    num_episodes = init_keys.shape[0]
    state = jax.vmap(env.init)(init_keys)
    game_length = jnp.zeros(num_episodes, dtype=jnp.int32)
    legal_sum = jnp.zeros(num_episodes, dtype=jnp.int32)

    def cond(carry):
        state, ply, *_ = carry
        return (~state.terminated.all()) & (ply < MAX_GAME_LENGTH)

    def body(carry):
        state, ply, game_length, legal_sum, key = carry
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, num_episodes)
        running = ~state.terminated  # games still in progress this ply
        actions = jax.vmap(_sample_legal_action)(step_keys, state.legal_action_mask)
        legal_count = state.legal_action_mask.sum(axis=-1).astype(jnp.int32)
        next_state = jax.vmap(lambda s, a: env.step(s, a))(state, actions)
        game_length = game_length + running.astype(jnp.int32)
        legal_sum = legal_sum + jnp.where(running, legal_count, 0)
        return next_state, ply + 1, game_length, legal_sum, key

    state, plies_run, game_length, legal_sum, _ = jax.lax.while_loop(
        cond, body, (state, jnp.int32(0), game_length, legal_sum, key)
    )
    black_points, white_points = jax.vmap(_score)(state._x.board)
    return {
        "game_length": game_length,
        "legal_sum": legal_sum,
        "plies_run": plies_run,
        "terminated": state.terminated,
        "consecutive_pass_count": state._x.consecutive_pass_count,
        "step_count": state._x.step_count,
        "black_points": black_points,
        "white_points": white_points,
    }


def analyze(num_episodes: int, seed: int):
    key = jax.random.PRNGKey(seed)
    init_key, run_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, num_episodes)

    # First call compiles; time a second identical call for throughput (compile excluded).
    res = jax.block_until_ready(_run_batch(run_key, init_keys))
    t0 = time.perf_counter()
    res = jax.block_until_ready(_run_batch(run_key, init_keys))
    elapsed = time.perf_counter() - t0

    gl = np.asarray(res["game_length"])
    ls = np.asarray(res["legal_sum"])
    plies_run = int(res["plies_run"])
    term = np.asarray(res["terminated"])
    cpc = np.asarray(res["consecutive_pass_count"])
    sc = np.asarray(res["step_count"])
    black_score = np.asarray(res["black_points"]).astype(np.float64)
    white_score = np.asarray(res["white_points"]).astype(np.float64) + KOMI

    black_win = black_score > white_score  # fractional komi => no draws
    margin = black_score - white_score  # > 0: black ahead

    # Terminal-reason inference from public state. Random-legal play never selects an
    # illegal action, so only the two natural reasons should appear (illegal is listed
    # for completeness and should stay at 0 here).
    two_pass = term & (cpc >= 2)
    max_length = term & ~two_pass & (sc >= MAX_GAME_LENGTH)
    unfinished = ~term  # would only appear if MAX_GAME_LENGTH were hit by the loop guard
    illegal = term & ~two_pass & ~max_length

    avg_legal = ls.sum() / max(gl.sum(), 1)
    env_steps = num_episodes * plies_run  # all envs step each ply (terminated ones no-op)

    print(f"layer_go random-legal self-play  (episodes={num_episodes}, seed={seed})")
    print("-" * 64)
    print(f"games                       : {num_episodes}")
    print(f"avg game length (plies)     : {gl.mean():.1f}")
    print(f"min / max game length       : {gl.min()} / {gl.max()}")
    print(f"black win rate              : {black_win.mean():.3f}")
    print(f"white win rate              : {(~black_win).mean():.3f}")
    print(f"avg final score margin (B-W): {margin.mean():+.2f}  (komi={KOMI})")
    print(f"avg legal actions / ply     : {avg_legal:.2f}  (of {env.num_actions}, incl. pass)")
    print("terminal reason:")
    print(f"  two consecutive passes    : {two_pass.mean():.3f}")
    print(f"  max game length ({MAX_GAME_LENGTH})    : {max_length.mean():.3f}")
    print(f"  illegal action            : {illegal.mean():.3f}  (expected 0 for legal-only play)")
    print(f"  unfinished                : {unfinished.mean():.3f}")
    print(f"pct reaching max length     : {100.0 * max_length.mean():.1f}%")
    print("-" * 64)
    print(f"throughput (vmap+jit)       : {env_steps / elapsed:,.0f} env-steps/s")
    print(f"  batch={num_episodes}, plies_run={plies_run}, wall={elapsed * 1e3:.1f} ms (compile excluded)")
    print("note: komi is untuned; random-play win rate is NOT a measure of game balance.")


def main():
    parser = argparse.ArgumentParser(description="Random-legal self-play sanity check for layer_go.")
    parser.add_argument("--episodes", type=int, default=64, help="number of games to play (batch size)")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()
    analyze(args.episodes, args.seed)


if __name__ == "__main__":
    main()
