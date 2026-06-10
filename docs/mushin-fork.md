# Mushin fork of pgx

This fork ([gweber/pgx](https://github.com/gweber/pgx), integration branch `mushin`) carries
performance and correctness work on top of upstream [sotetsuk/pgx](https://github.com/sotetsuk/pgx),
focused on chess and Go for AlphaZero-style self-play training (see the Mushin/ZenZero engine).

Branch layout: `main` mirrors upstream and is never committed to; `mushin` is the integration
branch consumed by training (pin a commit, bump deliberately); one feature branch per topic,
kept alive until upstreamed.

## Changes vs upstream

### Chess

| Change | Effect |
|---|---|
| Stockfish-style legal move generation: checkers, pin rays, king-danger squares computed once per position; no per-move make/unmake | `legal_action_mask` 14x faster |
| Occupancy bitboard (2x uint32) + `BETWEEN_MASK`: slider path-clear test is a 2-word AND | further ~1.3x on full step |
| **Bug fix**: the old `jnp.nonzero(..., size=200)` compaction silently dropped legal moves in positions with >200 pseudo-legal candidates (multi-queen self-play positions; the known 218-move position returned 200) | exact masks, no cap |
| **Bug fix**: `INIT_ZOBRIST_HASH` was a hardcoded constant that went stale when JAX's PRNG changed; repetitions involving the start position were counted one short | derived from the tables at import time ([upstream PR #1320](https://github.com/sotetsuk/pgx/pull/1320)) |
| **Bug fix**: `from_fen` inherited the startpos hash/board as phantom history ([upstream PR #1319](https://github.com/sotetsuk/pgx/pull/1319)) | clean history after FEN restore |
| `hash_history` 513 -> 101 entries (the 50-move rule bounds repetition lookback exactly) | -3.3 KB/state, 5x cheaper repetition scans |
| `board_history` as int8; `is_terminal` reuses the stored hash | -1.5 KB/state |
| Zobrist XOR-reduce built from sum/shift bit-parity (`xor_reduce`) | jax-metal / Apple GPU support ([upstream PR #1317](https://github.com/sotetsuk/pgx/pull/1317)) |

**Full `Game.step`, batch 2048, vmap+jit, CUDA (GB10): 8.30 ms -> 0.50 ms (16.7x).**
`GameState`: 11.1 KB -> 6.3 KB per env (matters when states are MCTS embeddings).

### Go

| Change | Effect |
|---|---|
| Territory flood fill skipped for non-terminal states (results were computed and discarded every step; under vmap the while_loop runs the batch worst case) | `rewards()` flat ~0.06 ms instead of 0.17-0.71 ms, 3-12x ([upstream PR #1321](https://github.com/sotetsuk/pgx/pull/1321)) |
| Chain stat accumulation via `segment_sum` instead of O(n^2) all-pairs comparison | `game.step` 1.32 -> 0.50 ms at batch 2048 (2.6x) |
| `board_history` as int8 | -8.7 KB/state (19x19: 18.8 KB -> 10.1 KB) |

### Misc

- `jax.random.categorical` axis fix for JAX >= 0.10 (cherry-picked upstream PR #1314)

## Validation methodology

- **Chess**: `tests/diff_vs_python_chess.py` — differential test against python-chess as ground
  truth: 17 adversarial FENs (kiwipete, perft 3-6, en-passant pins, castling through check,
  double check, the 218-move position) with full depth-2 expansion, plus 20 random games
  compared move-by-move. Forced threefold-repetition test (knight shuffle terminates at ply 8).
- **Go**: lockstep playouts (300 plies 19x19, 170 plies 9x9, batch 64) against the previous
  implementation: boards, legal masks, ko, rewards, terminal flags and observations
  bit-identical at every ply. Terminal scoring force-compared on dense boards.

Rejected experiments (benchmarked, then dropped): pointer-jumping CCL for territory scoring
(2-4x slower on realistic mid-game boards), `fori_loop` flood fill (always pays the worst
case), upstream `go-avoid-while` / `chess/bb` nibble packing.

## Benchmarking rules on shared boxes

One JAX process at a time, everything jitted, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_PYTHON_CLIENT_ALLOCATOR=platform`, modest batch sizes for compile-heavy graphs.
