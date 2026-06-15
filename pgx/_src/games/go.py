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

from typing import NamedTuple, Optional

import jax
from jax import Array, lax
from jax import numpy as jnp

from pgx._src.utils import bloom_insert, bloom_query, xor_reduce

ZOBRIST_BOARD = jax.random.randint(jax.random.PRNGKey(12345), (3, 19 * 19, 2), 0, 2**31 - 1, jnp.uint32)

# Tiered PSK: Bloom pre-filter (no false negatives) + exact recent window.
# Cascade: bloom "no" → definitely new, O(k=4); bloom "maybe" → O(K=16) exact check.
# State: 1 152 B vs 5 776 B for full hash_history (19×19); ~36× fewer PSK comparisons.
# Caveat: superko cycles > _PSK_RECENT_K moves apart are not detected — acceptable for RL.
_PSK_BLOOM_WORDS = 256  # 8 192-bit filter, 1 KB/state; FP ≈ 0.9 % for n=722 (19×19)
_PSK_RECENT_K = 16      # exact positions to keep; covers Ko and all typical short-range cycles


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)
    # ids of representative stone (smallest) in the connected stones
    board: Array = jnp.zeros(19 * 19, dtype=jnp.int16)  # b > 0, w < 0, empty = 0; chain ids fit int16
    board_history: Array = jnp.full((8, 19 * 19), 2, dtype=jnp.int8)  # for obs; values in {-1, 0, 1, 2}
    num_captured: Array = jnp.zeros(2, dtype=jnp.int32)  # (b, w)
    consecutive_pass_count: Array = jnp.int32(0)
    ko: Array = jnp.int32(-1)  # by SSK
    is_psk: Array = jnp.bool_(False)
    psk_bloom: Array = jnp.zeros(_PSK_BLOOM_WORDS, dtype=jnp.uint32)
    psk_recent: Array = jnp.zeros((_PSK_RECENT_K, 2), dtype=jnp.uint32)
    # cached _count(board): (num_pseudo, idx_sum, idx_squared_sum), refreshed in step().
    # legal_action_mask (board t+1) and the next _apply_action (same board) used to each
    # recompute it — caching halves the per-ply chain-stat work. Invariant: rebuild this
    # whenever board is modified outside step().
    chain_stats: Array = jnp.zeros((3, 19 * 19), dtype=jnp.int32)

    @property
    def color(self) -> Array:
        return self.step_count % 2


class Game:
    def __init__(
        self, size: int = 19, komi: float = 7.5, history_length: int = 8, max_termination_steps: Optional[int] = None
    ):
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = size * size * 2 if max_termination_steps is None else max_termination_steps
        # Precompute adjacency table once: shape (size**2, 4) int32.
        # Eliminates repeated _adj_ixs vmap in _count, legal_action_mask, and _count_ji.
        self.adj_mat = jax.vmap(_adj_ixs, in_axes=(0, None))(jnp.arange(size**2), size)

    def init(self) -> GameState:
        return GameState(
            board=jnp.zeros(self.size**2, dtype=jnp.int16),
            board_history=jnp.full((self.history_length, self.size**2), 2, dtype=jnp.int8),
            chain_stats=jnp.zeros((3, self.size**2), dtype=jnp.int32),  # _count of an empty board
        )

    def step(self, state: GameState, action: Array) -> GameState:
        state = state._replace(ko=jnp.int32(-1))
        # update state
        state = lax.cond(
            (action < self.size * self.size),
            lambda: _apply_action(state, action, self.size),
            lambda: _apply_pass(state),
        )
        # refresh the chain-stat cache for the new board (consumed by legal_action_mask
        # and by _apply_action on the next step)
        state = state._replace(chain_stats=jnp.stack(_count(state, self.size, self.adj_mat)))
        # update board history — circular buffer: write O(N) instead of rolling O(history×N)
        hist_slot = state.step_count % self.history_length
        board_history = state.board_history.at[hist_slot].set(
            jnp.clip(state.board, -1, 1).astype(jnp.int8)
        )
        state = state._replace(board_history=board_history)
        # check PSK: tiered Bloom pre-filter + exact recent window
        # bloom "no" → O(k) definitely-not-seen; bloom "maybe" → O(K) exact recent check.
        hash_ = _compute_hash(state)
        bloom_hit = bloom_query(state.psk_bloom, hash_)
        recent_hit = (hash_ == state.psk_recent).all(axis=-1).any()
        is_psk = (state.consecutive_pass_count == 0) & bloom_hit & recent_hit
        psk_slot = state.step_count % _PSK_RECENT_K
        new_bloom = bloom_insert(state.psk_bloom, hash_)
        new_recent = state.psk_recent.at[psk_slot].set(hash_)
        state = state._replace(psk_bloom=new_bloom, psk_recent=new_recent, is_psk=is_psk)
        # increment turns
        state = state._replace(step_count=state.step_count + 1)
        return state

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        my_sign, _ = _signs(color)

        def _make(i):
            c = jnp.int32([1, -1])[i % 2] * my_sign
            # circular buffer: most-recent slot = (step_count-1) % history_length
            slot = (state.step_count - 1 - (i // 2)) % self.history_length
            return state.board_history[slot] == c

        log = jax.vmap(_make)(jnp.arange(self.history_length * 2))
        color = jnp.full_like(log[0], color)  # b = 0, w = 1
        return jnp.vstack([log, color]).transpose().reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Array:
        # some logic is inspired by OpenSpiel's Go implementation
        is_empty = state.board == 0
        my_sign, opp_sign = _signs(state.color)
        num_pseudo, idx_sum, idx_squared_sum = state.chain_stats
        chain_ix = jnp.abs(state.board) - 1
        in_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
        has_liberty = (state.board * my_sign > 0) & ~in_atari
        can_kill = (state.board * opp_sign > 0) & in_atari

        adj_mat = self.adj_mat  # (size**2, 4) precomputed
        on_board = adj_mat != -1  # (size**2, 4) bool, static shape

        # Fully vectorised: replace per-cell vmap with a single broadcast over adj_mat.
        # safe_adj clamps -1 sentinel to 0 so out-of-board neighbors index safely;
        # those entries are masked out by on_board before the .any().
        safe_adj = jnp.where(on_board, adj_mat, 0)  # clamp -1 → 0 for safe gather
        # One gather instead of three: OR the three (N,) conditions first.
        neighbor_ok = is_empty | can_kill | has_liberty  # (N,)
        ok = on_board & neighbor_ok[safe_adj]  # (N, 4)
        mask = is_empty & ok.any(axis=1)
        mask = lax.select(state.ko == -1, mask, mask.at[state.ko].set(False))
        return jnp.append(mask, True)  # pass is always legal

    def is_terminal(self, state: GameState) -> Array:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Array:
        is_terminal = self.is_terminal(state)
        scores = _count_scores(state, self.size, self.adj_mat, enable=is_terminal)
        is_black_win = scores[0] - self.komi > scores[1]
        rewards = lax.select(is_black_win, jnp.float32([1, -1]), jnp.float32([-1, 1]))
        to_play = state.color
        rewards = lax.select(state.is_psk, jnp.float32([-1, -1]).at[to_play].set(1.0), rewards)
        rewards = lax.select(is_terminal, rewards, jnp.zeros(2, dtype=jnp.float32))
        return rewards


def _apply_pass(state: GameState) -> GameState:
    return state._replace(consecutive_pass_count=state.consecutive_pass_count + 1)


def _apply_action(state: GameState, action, size) -> GameState:
    state = state._replace(consecutive_pass_count=0)
    my_sign, opp_sign = _signs(state.color)

    # remove killed stones
    adj_ixs = _adj_ixs(action, size)
    adj_ids = state.board[adj_ixs]
    num_pseudo, idx_sum, idx_squared_sum = state.chain_stats
    chain_ix = jnp.abs(adj_ids) - 1
    is_atari = (idx_sum[chain_ix] ** 2) == idx_squared_sum[chain_ix] * num_pseudo[chain_ix]
    # In atari there is exactly one DISTINCT liberty L, but it may be adjacent to several stones of
    # the chain, so idx_sum == num_pseudo * (L + 1) (not just L + 1). Divide by num_pseudo to recover
    # L. (The previous `idx_sum - 1` assumed num_pseudo == 1 and missed captures of chains whose
    # single liberty touches more than one of their stones, e.g. an L-shaped 3-stone chain.)
    single_liberty = idx_sum[chain_ix] // jnp.maximum(num_pseudo[chain_ix], 1) - 1
    is_killed = (adj_ixs != -1) & (adj_ids * opp_sign > 0) & is_atari & (single_liberty == action)
    surrounded_stones = (state.board[:, None] == adj_ids) & (is_killed[None, :])
    num_captured = jnp.count_nonzero(surrounded_stones)
    ko_ix = jnp.nonzero(is_killed, size=1)[0][0]
    ko_may_occur = ((adj_ixs == -1) | (state.board[adj_ixs] * opp_sign > 0)).all()
    state = state._replace(
        board=jnp.where(surrounded_stones.any(axis=-1), 0, state.board),
        num_captured=state.num_captured.at[state.color].add(num_captured),
        ko=lax.select(ko_may_occur & (num_captured == 1), adj_ixs[ko_ix], -1),
    )

    # set stone
    state = state._replace(board=state.board.at[action].set(((action + 1) * my_sign).astype(state.board.dtype)))

    # merge adjacent chains
    is_my_chain = state.board[adj_ixs] * my_sign > 0
    should_merge = (adj_ixs != -1) & is_my_chain
    new_id = state.board[action]
    tgt_ids = state.board[adj_ixs]
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), 9999))
    smallest_id = jnp.minimum(jnp.abs(new_id), smallest_id) * my_sign
    mask = (state.board == new_id) | (should_merge[None, :] & (state.board[:, None] == tgt_ids[None, :])).any(axis=-1)
    state = state._replace(board=jnp.where(mask, smallest_id, state.board))

    return state


def _count(state: GameState, size, adj_mat):
    board = jnp.abs(state.board)
    is_empty = board == 0

    # Pack all three per-cell neighbor stats into one (N,3) gather + one segment_sum.
    # Single kernel vs. three separate ones → less launch overhead, better coalescing.
    on_board = adj_mat != -1                                              # (N, 4)
    safe_adj = jnp.where(on_board, adj_mat, 0)                           # clamp -1 safe
    # The pseudo-liberty moments (idx_sum, idx_sq_sum) must be taken over a chain's EMPTY
    # neighbours (its liberties) only; the atari identity idx_sum**2 == num_pseudo * idx_sq_sum
    # relies on that. idx1 / idx1**2 are therefore masked by is_empty (a stone neighbour
    # contributes 0). Without the mask, stone-neighbour indices leak into the moments and the
    # identity misfires, so a chain touching another stone is not detected as being in atari and
    # its capture is missed (e.g. a 1-liberty corner stone next to an enemy stone).
    e = is_empty.astype(jnp.int32)
    idx1 = jnp.arange(1, size**2 + 1, dtype=jnp.int32) * e
    vals = jnp.stack([e, idx1, idx1 * idx1], axis=1)  # (N, 3): empty-neighbour count, idx sum, idx^2 sum
    nb = jnp.where(on_board[:, :, None], vals[safe_adj], 0).sum(axis=1) # (N, 3)

    # scatter-add per-point stats into their chains; empties go into an overflow bucket
    seg = jnp.where(is_empty, size**2, board.astype(jnp.int32) - 1)
    result = jax.ops.segment_sum(nb, seg, num_segments=size**2 + 1)[: size**2]  # (N, 3)
    return result[:, 0], result[:, 1], result[:, 2]


def _signs(color):
    return jnp.int16([[1, -1], [-1, 1]])[color]  # (my_sign, opp_sign)


def _adj_ixs(xy, size):
    dx, dy = jnp.int32([-1, +1, 0, 0]), jnp.int32([0, 0, -1, +1])
    xs, ys = xy // size + dx, xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)  # -1 if out of board


def _compute_hash(state: GameState):
    board = jnp.clip(state.board, -1, 1)
    to_reduce = ZOBRIST_BOARD[board, jnp.arange(board.shape[-1])]
    return xor_reduce(to_reduce, 0)


def _count_scores(state: GameState, size, adj_mat, enable=True):
    # `enable=False` replaces the board with a fully-occupied dummy whose flood fill converges
    # immediately. rewards() discards the scores of non-terminal states anyway, but under
    # vmap/jit the while_loop below runs as many rounds as the worst board in the batch needs —
    # without the dummy, every step pays the full territory fill even though almost no state
    # in the batch is terminal (the empty early-game board is the worst case at ~2*size rounds).
    def calc_point(c):
        return _count_ji(state, c, size, adj_mat, enable) + jnp.count_nonzero(state.board * c > 0)

    return jax.vmap(calc_point)(jnp.int32([1, -1]))


def _count_ji(state: GameState, color: int, size: int, adj_mat, enable=True):
    board = jnp.clip(state.board * color, -1, 1)  # my stone: 1, opp stone: -1
    board = jnp.where(enable, board, 1)
    # adj_mat: (size**2, 4) precomputed adjacency; -1 means off-board
    on_board = adj_mat != -1
    safe_adj = jnp.where(on_board, adj_mat, 0)  # clamp for safe gather

    def fill_opp(x):
        b, _ = x
        # true if empty and adjacent to opponent's stone
        mask = (b == 0) & (on_board & (b[safe_adj] == -1)).any(axis=1)
        return jnp.where(mask, -1, b), mask.any()

    board, _ = lax.while_loop(lambda x: x[1], fill_opp, (board, True))
    return (board == 0).sum()
