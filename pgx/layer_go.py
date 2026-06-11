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
"""Layer Go -- a compact 3D Go variant played on a 5x5x3 orthogonal grid.

Points are addressed by ``(x, y, z)`` with ``x, y in [0, 4]`` and ``z in [0, 2]``.
The linear index of a point is ``z * WIDTH * HEIGHT + y * WIDTH + x``. Adjacency
is orthogonal only (up to 6 neighbours per point); diagonal touching never
connects stones nor grants liberties. See ``docs/layer_go.md`` for the full rules.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

WIDTH = 5
HEIGHT = 5
DEPTH = 3
BOARD_SIZE = WIDTH * HEIGHT * DEPTH  # 75
PASS_ACTION = BOARD_SIZE  # 75
ACTION_SIZE = BOARD_SIZE + 1  # 76
KOMI = 7.5

# Robust finite-episode bound: a game is force-terminated (and area-scored) once it
# reaches MAX_GAME_LENGTH plies. The superko history holds one absolute board per stone
# move plus the initial empty board, so MAX_HISTORY = MAX_GAME_LENGTH + 1 rows suffice.
MAX_GAME_LENGTH = 512
MAX_HISTORY = MAX_GAME_LENGTH + 1  # 513


def coord_to_index(x: int, y: int, z: int) -> int:
    """Map a ``(x, y, z)`` coordinate to its linear board index."""
    return z * WIDTH * HEIGHT + y * WIDTH + x


def index_to_coord(index: int):
    """Map a linear board index back to its ``(x, y, z)`` coordinate."""
    z = index // (WIDTH * HEIGHT)
    rem = index % (WIDTH * HEIGHT)
    y = rem // WIDTH
    x = rem % WIDTH
    return x, y, z


def _build_neighbor_table():
    """Precompute the fixed ``(75, 6)`` orthogonal neighbour table (-1 = off-board)."""
    table = []
    deltas = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for index in range(BOARD_SIZE):
        x, y, z = index_to_coord(index)
        row = []
        for dx, dy, dz in deltas:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and 0 <= nz < DEPTH:
                row.append(coord_to_index(nx, ny, nz))
            else:
                row.append(-1)
        table.append(row)
    return table


# Static neighbour tables (computed once at import; used inside jitted functions).
NEIGHBORS = jnp.array(_build_neighbor_table(), dtype=jnp.int32)  # (75, 6), -1 if off-board
_ON_BOARD = NEIGHBORS >= 0  # (75, 6) bool
_SAFE_NEIGHBORS = jnp.where(_ON_BOARD, NEIGHBORS, 0)  # off-board entries point to 0 (masked out)

# Zobrist table for positional-superko board hashing. Indexed by [stone, point, word]:
# row 0 = empty, row 1 = black (+1), row 2 = white (reached by board value -1 via the usual
# negative-index wrap, matching pgx Go). Two uint32 words give a 64-bit hash per board.
_ZOBRIST = jax.random.randint(jax.random.PRNGKey(20240611), (3, BOARD_SIZE, 2), 0, 2**31 - 1, jnp.uint32)


def _board_hash(board: Array) -> Array:
    """64-bit (``(2,)`` uint32) Zobrist-style hash of an absolute board (values in {-1, 0, 1}).

    Uses an *additive* (mod 2**32) reduction of two independent random tables rather than an
    XOR reduction: it is just as collision-resistant for this use (~2**-64 per pair) but avoids
    the bit-parity expansion of ``xor_reduce``, and ``sum`` legalizes on every XLA backend
    (including Apple ``jax-metal``).
    """
    contributions = _ZOBRIST[jnp.clip(board, -1, 1), jnp.arange(BOARD_SIZE)]  # (75, 2); -1 -> row 2
    return jnp.sum(contributions, axis=0)  # additive hash over points -> (2,), wraps mod 2**32


_EMPTY_BOARD_HASH = _board_hash(jnp.zeros(BOARD_SIZE, dtype=jnp.int8))
# Initial superko history: the empty board's hash at row 0; the rest are unused (validity is
# tracked by `num_history`, so the unused rows' contents are never compared).
_EMPTY_HASH_HISTORY = jnp.zeros((MAX_HISTORY, 2), dtype=jnp.uint32).at[0].set(_EMPTY_BOARD_HASH)


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)  # ply count; color = step_count % 2 (0 = black, 1 = white)
    board: Array = jnp.zeros(BOARD_SIZE, dtype=jnp.int8)  # 0 = empty, +1 = black, -1 = white
    consecutive_pass_count: Array = jnp.int32(0)
    # Positional-superko history: a Zobrist hash (2x uint32) of the initial empty board and of
    # every board reached by a stone move. Passes do not change the board, so they add no row.
    # `num_history` is the count of valid rows / the next write index; only the first
    # `num_history` rows are ever compared, so unused rows need no sentinel.
    hash_history: Array = _EMPTY_HASH_HISTORY
    num_history: Array = jnp.int32(1)

    @property
    def color(self) -> Array:
        return self.step_count % 2


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((DEPTH, HEIGHT, WIDTH, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(ACTION_SIZE, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _player_order: Array = jnp.int32([0, 1])  # color -> player id; [0, 1] or [1, 0]
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "layer_go"


class LayerGo(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "layer_go"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _color_to_sign(color: Array) -> Array:
    """color 0 (black) -> +1, color 1 (white) -> -1."""
    return jnp.where(color == 0, 1, -1).astype(jnp.int8)


def _group_alive(board: Array, stones: Array) -> Array:
    """Return a bool[75] flagging every stone in ``stones`` whose group has >= 1 liberty.

    A liberty is an empty point orthogonally adjacent to any stone of the group. Aliveness
    is seeded at stones that directly touch an empty point and then flooded through
    same-colour orthogonal adjacency. The ``while_loop`` runs until the fixpoint is reached
    (a few iterations for realistic groups; bounded by the longest in-group path), instead of
    always paying ``BOARD_SIZE`` iterations.
    """
    empty = board == 0
    has_empty_neighbor = (empty[_SAFE_NEIGHBORS] & _ON_BOARD).any(axis=1)
    seed = stones & has_empty_neighbor

    def cond(carry):
        _, changed = carry
        return changed

    def body(carry):
        alive, _ = carry
        neighbor_alive = (alive[_SAFE_NEIGHBORS] & _ON_BOARD).any(axis=1)
        new_alive = stones & (alive | neighbor_alive)
        return new_alive, (new_alive != alive).any()

    alive, _ = jax.lax.while_loop(cond, body, (seed, jnp.bool_(True)))
    return alive


def _resulting_board(board: Array, my_sign: Array, action: Array):
    """Place ``my_sign`` at ``action`` then remove opponent groups left with no liberty.

    Returns ``(new_board, num_captured)``. Out-of-bounds ``action`` (e.g. the pass action)
    leaves the board unchanged because JAX drops out-of-bounds scatter updates.
    """
    placed = board.at[action].set(my_sign)
    opp_sign = -my_sign
    opp_stones = placed == opp_sign
    opp_alive = _group_alive(placed, opp_stones)
    captured = opp_stones & ~opp_alive
    new_board = jnp.where(captured, 0, placed)
    return new_board, jnp.count_nonzero(captured)


def _is_superko(hash_history: Array, num_history: Array, candidate_hash: Array) -> Array:
    """True if ``candidate_hash`` equals any board hash already recorded in the game.

    This is **positional** superko: only the stone arrangement is hashed, never the player to
    move. Only the first ``num_history`` rows are valid, so the comparison is masked to them.
    Hashing makes this a ``num_history``-wide compare of 64-bit values instead of a full
    board-by-board scan; the (vanishingly small, ~2^-64 per pair) collision risk is documented
    in ``docs/layer_go.md``.
    """
    valid = jnp.arange(MAX_HISTORY) < num_history
    same = jnp.all(hash_history == candidate_hash, axis=1)
    return (same & valid).any()


def _is_legal_point(board: Array, my_sign: Array, action: Array, hash_history: Array, num_history: Array) -> Array:
    """A point action is legal iff the point is empty, the placement is not suicide (after
    captures), and the resulting board has not appeared before in the game (positional superko).
    """
    is_empty = board[action] == 0
    new_board, _ = _resulting_board(board, my_sign, action)
    my_alive = _group_alive(new_board, new_board == my_sign)
    not_suicide = my_alive[action]
    not_superko = ~_is_superko(hash_history, num_history, _board_hash(new_board))
    return is_empty & not_suicide & not_superko


def _legal_action_mask(board: Array, color: Array, hash_history: Array, num_history: Array) -> Array:
    my_sign = _color_to_sign(color)
    point_mask = jax.vmap(lambda a: _is_legal_point(board, my_sign, a, hash_history, num_history))(
        jnp.arange(BOARD_SIZE)
    )
    return jnp.append(point_mask, TRUE)  # pass is always legal (no superko check on pass)


def _territory(board: Array):
    """Area-scoring territory via empty-region flood fill.

    Returns ``(black_territory, white_territory)`` bool[75] masks. An empty region belongs
    to a colour iff every stone bordering the region is that colour and at least one
    bordering stone exists; otherwise the region is neutral.
    """
    empty = board == 0
    touches_black_seed = empty & ((board[_SAFE_NEIGHBORS] == 1) & _ON_BOARD).any(axis=1)
    touches_white_seed = empty & ((board[_SAFE_NEIGHBORS] == -1) & _ON_BOARD).any(axis=1)

    def body(_, carry):
        tb, tw = carry
        tb = empty & (tb | (tb[_SAFE_NEIGHBORS] & _ON_BOARD).any(axis=1))
        tw = empty & (tw | (tw[_SAFE_NEIGHBORS] & _ON_BOARD).any(axis=1))
        return tb, tw

    touches_black, touches_white = jax.lax.fori_loop(0, BOARD_SIZE, body, (touches_black_seed, touches_white_seed))
    black_territory = empty & touches_black & ~touches_white
    white_territory = empty & touches_white & ~touches_black
    return black_territory, white_territory


def _score(board: Array):
    """Return area scores ``(black_points, white_points)`` excluding komi."""
    black_territory, white_territory = _territory(board)
    black_points = jnp.count_nonzero(board == 1) + jnp.count_nonzero(black_territory)
    white_points = jnp.count_nonzero(board == -1) + jnp.count_nonzero(white_territory)
    return black_points, white_points


def _rewards(board: Array, player_order: Array, terminated: Array) -> Array:
    black_points, white_points = _score(board)
    black_wins = black_points > white_points + KOMI
    # Indexed by color: [black_reward, white_reward].
    rewards_by_color = jnp.where(black_wins, jnp.float32([1.0, -1.0]), jnp.float32([-1.0, 1.0]))
    rewards = rewards_by_color[player_order]  # reindex to player id
    return jnp.where(terminated, rewards, jnp.zeros(2, dtype=jnp.float32))


def _init(key: PRNGKey) -> State:
    player_order = jnp.array([[0, 1], [1, 0]])[jax.random.bernoulli(key).astype(jnp.int32)]
    x = GameState()  # default GameState already records the empty board's hash (row 0)
    return State(  # type: ignore
        current_player=player_order[0],
        legal_action_mask=_legal_action_mask(x.board, jnp.int32(0), x.hash_history, x.num_history),
        _player_order=player_order,
        _x=x,
    )


def _step(state: State, action: Array) -> State:
    x = state._x
    is_pass = action == PASS_ACTION
    my_sign = _color_to_sign(x.color)

    placed_board, _ = _resulting_board(x.board, my_sign, action)
    new_board = jnp.where(is_pass, x.board, placed_board).astype(jnp.int8)
    consecutive_pass_count = jnp.where(is_pass, x.consecutive_pass_count + 1, 0).astype(jnp.int32)

    # Record the new board's hash for stone moves only. A pass leaves the board unchanged, so
    # it neither advances nor duplicates the superko history (and never becomes illegal).
    recorded_history = x.hash_history.at[x.num_history].set(_board_hash(new_board))
    hash_history = jnp.where(is_pass, x.hash_history, recorded_history)
    num_history = jnp.where(is_pass, x.num_history, x.num_history + 1).astype(jnp.int32)

    next_x = GameState(
        step_count=x.step_count + 1,
        board=new_board,
        consecutive_pass_count=consecutive_pass_count,
        hash_history=hash_history,
        num_history=num_history,
    )
    two_consecutive_passes = consecutive_pass_count >= 2
    max_length_reached = next_x.step_count >= MAX_GAME_LENGTH
    terminated = two_consecutive_passes | max_length_reached
    rewards = _rewards(new_board, state._player_order, terminated)

    return state.replace(  # type: ignore
        current_player=state._player_order[next_x.color],
        legal_action_mask=_legal_action_mask(new_board, next_x.color, hash_history, num_history),
        rewards=rewards,
        terminated=terminated,
        _x=next_x,
    )


def _observe(state: State, player_id: Array) -> Array:
    """Current-player-relative planes, shape ``(DEPTH, HEIGHT, WIDTH, 2)``.

    Channel 0 = the observing player's stones, channel 1 = the opponent's stones.
    """
    x = state._x
    my_color = jax.lax.select(player_id == state.current_player, x.color, 1 - x.color)
    my_sign = _color_to_sign(my_color)
    board = x.board * my_sign  # +1 = my stone, -1 = opponent stone
    my_plane = board > 0
    opp_plane = board < 0
    obs = jnp.stack([my_plane, opp_plane], axis=-1)  # (75, 2)
    return obs.reshape((DEPTH, HEIGHT, WIDTH, 2))
