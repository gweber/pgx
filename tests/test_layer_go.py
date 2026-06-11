import jax
import jax.numpy as jnp

import pgx
from pgx.layer_go import (
    ACTION_SIZE,
    BOARD_SIZE,
    DEPTH,
    HEIGHT,
    KOMI,
    MAX_GAME_LENGTH,
    MAX_HISTORY,
    NEIGHBORS,
    PASS_ACTION,
    WIDTH,
    GameState,
    LayerGo,
    State,
    _board_hash,
    _color_to_sign,
    _group_alive,
    _is_superko,
    _legal_action_mask,
    _resulting_board,
    _score,
    coord_to_index,
    index_to_coord,
)

env = LayerGo()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def _history_from_boards(boards):
    """Build a (MAX_HISTORY, 2) Zobrist superko history seeded with `boards` (in order)."""
    hist = jnp.zeros((MAX_HISTORY, 2), dtype=jnp.uint32)
    for i, b in enumerate(boards):
        hist = hist.at[i].set(_board_hash(jnp.int8(b)))
    return hist, jnp.int32(len(boards))


def _empty_state(board, color=0, player_order=(0, 1), history_boards=None, step_count=None):
    """Build a State with a hand-crafted board and the given color to move.

    `history_boards` seeds the positional-superko (hash) history (defaults to just the empty
    board, matching a fresh GameState). `step_count` overrides the ply counter (its parity also
    sets the color unless `color` already encodes it).
    """
    if history_boards is None:
        hash_history, num_history = _history_from_boards([[0] * BOARD_SIZE])
    else:
        hash_history, num_history = _history_from_boards(history_boards)
    sc = color if step_count is None else step_count
    x = GameState(
        step_count=jnp.int32(sc),
        board=jnp.int8(board),
        hash_history=hash_history,
        num_history=num_history,
    )
    return State(  # type: ignore
        current_player=jnp.int32(player_order[int(sc) % 2]),
        legal_action_mask=_legal_action_mask(jnp.int8(board), jnp.int32(int(sc) % 2), hash_history, num_history),
        _player_order=jnp.int32(player_order),
        _x=x,
    )


# --------------------------------------------------------------------------- #
# Coordinate / index tests
# --------------------------------------------------------------------------- #
def test_constants():
    assert WIDTH == 5 and HEIGHT == 5 and DEPTH == 3
    assert BOARD_SIZE == 75
    assert PASS_ACTION == 75
    assert ACTION_SIZE == 76
    assert KOMI == 7.5


def test_specific_indices():
    assert coord_to_index(0, 0, 0) == 0
    assert coord_to_index(4, 0, 0) == 4
    assert coord_to_index(0, 1, 0) == 5
    assert coord_to_index(0, 0, 1) == 25
    assert coord_to_index(4, 4, 2) == 74


def test_index_coord_roundtrip():
    for index in range(BOARD_SIZE):
        x, y, z = index_to_coord(index)
        assert coord_to_index(x, y, z) == index
    for z in range(DEPTH):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                index = coord_to_index(x, y, z)
                assert index_to_coord(index) == (x, y, z)


# --------------------------------------------------------------------------- #
# Neighbor tests
# --------------------------------------------------------------------------- #
def _neighbors_of(x, y, z):
    row = NEIGHBORS[coord_to_index(x, y, z)]
    return set(int(v) for v in row if int(v) >= 0)


def test_neighbor_counts():
    assert len(_neighbors_of(0, 0, 0)) == 3  # corner
    assert len(_neighbors_of(4, 4, 2)) == 3  # opposite corner
    assert len(_neighbors_of(2, 2, 0)) == 5  # face center on outer layer
    assert len(_neighbors_of(2, 2, 1)) == 6  # fully interior
    assert len(_neighbors_of(2, 0, 0)) == 4  # edge, non-corner


def test_diagonal_not_neighbor():
    n000 = _neighbors_of(0, 0, 0)
    # diagonal in-plane is not adjacent
    assert coord_to_index(1, 1, 0) not in n000
    # diagonal across layers is not adjacent
    assert coord_to_index(1, 0, 1) not in n000
    # orthogonal up IS adjacent
    assert coord_to_index(0, 0, 1) in n000
    # orthogonal in-plane neighbors ARE adjacent
    assert coord_to_index(1, 0, 0) in n000
    assert coord_to_index(0, 1, 0) in n000


def test_neighbor_table_symmetry():
    for i in range(BOARD_SIZE):
        for j in NEIGHBORS[i]:
            j = int(j)
            if j >= 0:
                assert i in set(int(v) for v in NEIGHBORS[j] if int(v) >= 0)


# --------------------------------------------------------------------------- #
# Basic move tests
# --------------------------------------------------------------------------- #
def test_init_empty_board():
    state = init(jax.random.PRNGKey(0))
    assert (state._x.board == 0).all()
    assert state._x.step_count == 0
    assert not state.terminated
    assert (state.rewards == 0).all()


def test_place_and_alternation():
    state = init(jax.random.PRNGKey(0))
    first_player = int(state.current_player)
    state = step(state, jnp.int32(coord_to_index(2, 2, 1)))
    assert state._x.board[coord_to_index(2, 2, 1)] == 1  # black stone placed
    assert int(state.current_player) != first_player
    state = step(state, jnp.int32(coord_to_index(0, 0, 0)))
    assert state._x.board[coord_to_index(0, 0, 0)] == -1  # white stone placed
    assert int(state.current_player) == first_player


def test_occupied_is_illegal():
    state = init(jax.random.PRNGKey(0))
    a = coord_to_index(2, 2, 1)
    state = step(state, jnp.int32(a))
    assert not bool(state.legal_action_mask[a])


def test_pass_is_legal_and_two_passes_terminate():
    state = init(jax.random.PRNGKey(0))
    assert bool(state.legal_action_mask[PASS_ACTION])
    state = step(state, jnp.int32(PASS_ACTION))
    assert not state.terminated
    assert bool(state.legal_action_mask[PASS_ACTION])
    state = step(state, jnp.int32(PASS_ACTION))
    assert bool(state.terminated)


# --------------------------------------------------------------------------- #
# Capture tests
# --------------------------------------------------------------------------- #
def test_simple_2d_capture():
    # White stone at (1,1,0) surrounded by black on its 4 in-plane neighbors and capped
    # above by black; the only remaining liberty is below — but z=0 has no below, so the
    # 5 orthogonal neighbors that exist must all be black to capture.
    board = [0] * BOARD_SIZE
    white = coord_to_index(1, 1, 0)
    board[white] = -1
    # neighbors of (1,1,0): (0,1,0),(2,1,0),(1,0,0),(1,2,0),(1,1,1)
    surround = [
        coord_to_index(0, 1, 0),
        coord_to_index(2, 1, 0),
        coord_to_index(1, 0, 0),
        coord_to_index(1, 2, 0),
    ]
    for s in surround:
        board[s] = 1
    # last liberty is (1,1,1) above; black to move fills it -> capture
    state = _empty_state(board, color=0)
    last = coord_to_index(1, 1, 1)
    assert bool(state.legal_action_mask[last])
    state = step(state, jnp.int32(last))
    assert state._x.board[white] == 0  # captured


def test_z_axis_liberty_prevents_capture():
    # Black group on layer 0 fully surrounded in-plane and from below (no below at z=0),
    # but with an open liberty straight above on layer 1 -> not captured.
    board = [0] * BOARD_SIZE
    black = coord_to_index(2, 2, 0)
    board[black] = 1
    for s in [coord_to_index(1, 2, 0), coord_to_index(3, 2, 0), coord_to_index(2, 1, 0), coord_to_index(2, 3, 0)]:
        board[s] = -1
    # white to move; the only empty point adjacent to the black stone is (2,2,1) above.
    # Filling it is the capture move (it removes black's last liberty).
    state = _empty_state(board, color=1)  # white to move
    above = coord_to_index(2, 2, 1)
    # white playing above should capture (fills last liberty of single black stone)
    assert bool(state.legal_action_mask[above])
    after = step(state, jnp.int32(above))
    assert after._x.board[black] == 0


def test_z_axis_capture():
    # Single black stone at interior (2,2,1) with 6 neighbors. White occupies 5 of them;
    # the 6th (below) is white's capturing move filling the last liberty.
    board = [0] * BOARD_SIZE
    black = coord_to_index(2, 2, 1)
    board[black] = 1
    neighbors = [int(v) for v in NEIGHBORS[black]]
    below = coord_to_index(2, 2, 0)
    for n in neighbors:
        if n != below:
            board[n] = -1
    state = _empty_state(board, color=1)  # white to move
    # Black currently still has the 'below' liberty -> not yet captured.
    assert jnp.int32(board)[black] == 1
    state = step(state, jnp.int32(below))
    assert state._x.board[black] == 0  # captured from below (z-axis capture)


def test_diagonal_does_not_connect():
    # Two black stones touching only diagonally in a plane are separate groups, and a
    # diagonal neighbor does not provide a liberty.
    board = [0] * BOARD_SIZE
    a = coord_to_index(1, 1, 0)
    b = coord_to_index(2, 2, 0)  # diagonal to a
    board[a] = 1
    board[b] = 1
    # Surround stone `a` on all 5 of its existing orthogonal neighbors with white.
    a_neighbors = [int(v) for v in NEIGHBORS[a] if int(v) >= 0]
    # leave one neighbor empty to be the capturing move
    capture_move = a_neighbors[0]
    for n in a_neighbors[1:]:
        board[n] = -1
    state = _empty_state(board, color=1)  # white to move
    state = step(state, jnp.int32(capture_move))
    # `a` captured (diagonal `b` gave it no liberty / no connection)
    assert state._x.board[a] == 0
    # `b` is untouched: diagonal stone was a separate group
    assert state._x.board[b] == 1


def test_multi_stone_group_capture_across_layers():
    # A 2-stone black group spanning two layers: (2,2,0) and (2,2,1), connected via z.
    # White surrounds every orthogonal liberty; the last fill captures BOTH stones.
    board = [0] * BOARD_SIZE
    g0 = coord_to_index(2, 2, 0)
    g1 = coord_to_index(2, 2, 1)
    board[g0] = 1
    board[g1] = 1
    group = {g0, g1}
    liberties = []
    for stone in (g0, g1):
        for n in NEIGHBORS[stone]:
            n = int(n)
            if n >= 0 and n not in group:
                liberties.append(n)
    liberties = list(dict.fromkeys(liberties))  # unique, preserve order
    capture_move = liberties[0]
    for n in liberties[1:]:
        board[n] = -1
    state = _empty_state(board, color=1)  # white to move
    state = step(state, jnp.int32(capture_move))
    assert state._x.board[g0] == 0
    assert state._x.board[g1] == 0


# --------------------------------------------------------------------------- #
# Suicide tests
# --------------------------------------------------------------------------- #
def test_pure_suicide_is_illegal():
    # An empty point whose every orthogonal neighbor is white: black playing there is suicide.
    board = [0] * BOARD_SIZE
    target = coord_to_index(0, 0, 0)
    for n in NEIGHBORS[target]:
        n = int(n)
        if n >= 0:
            board[n] = -1
    state = _empty_state(board, color=0)  # black to move
    assert not bool(state.legal_action_mask[target])


def test_capture_then_self_liberty_is_legal():
    # Black plays the point that fills its own would-be last liberty, but doing so captures
    # an adjacent white stone in atari, which frees a liberty -> the move is legal.
    board = [0] * BOARD_SIZE
    target = coord_to_index(0, 0, 0)  # corner, 3 neighbors
    n = [int(v) for v in NEIGHBORS[target] if int(v) >= 0]
    # two neighbors are black (so target's group would be surrounded), one is a white stone in atari
    white = n[0]
    board[n[1]] = 1
    board[n[2]] = 1
    board[white] = -1
    # Surround the white stone's other liberties so playing `target` captures it.
    for w_n in NEIGHBORS[white]:
        w_n = int(w_n)
        if w_n >= 0 and w_n != target and board[w_n] == 0:
            board[w_n] = 1
    state = _empty_state(board, color=0)  # black to move
    assert bool(state.legal_action_mask[target])
    state = step(state, jnp.int32(target))
    assert state._x.board[target] == 1  # black stone stayed (legal)
    assert state._x.board[white] == 0  # white captured


# --------------------------------------------------------------------------- #
# Legal mask tests
# --------------------------------------------------------------------------- #
def test_initial_legal_mask_all_true():
    state = init(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (ACTION_SIZE,)
    assert bool(state.legal_action_mask.all())


def test_legal_mask_after_placement():
    state = init(jax.random.PRNGKey(0))
    a = coord_to_index(3, 1, 2)
    state = step(state, jnp.int32(a))
    assert not bool(state.legal_action_mask[a])  # occupied
    assert bool(state.legal_action_mask[PASS_ACTION])  # pass still legal


# --------------------------------------------------------------------------- #
# Scoring tests
# --------------------------------------------------------------------------- #
def test_score_black_controls_region():
    # A single black stone -> the whole empty board is black territory.
    board = [0] * BOARD_SIZE
    board[coord_to_index(2, 2, 1)] = 1
    black_pts, white_pts = _score(jnp.int32(board))
    assert int(black_pts) == BOARD_SIZE  # 1 stone + 74 territory
    assert int(white_pts) == 0


def test_score_white_controls_region():
    board = [0] * BOARD_SIZE
    board[coord_to_index(2, 2, 1)] = -1
    black_pts, white_pts = _score(jnp.int32(board))
    assert int(white_pts) == BOARD_SIZE
    assert int(black_pts) == 0


def test_score_mixed_border_region_is_neutral():
    # One black and one (non-adjacent, non-diagonal-issue) white stone share the empty region
    # border -> region neutral; each scores only its own stone.
    board = [0] * BOARD_SIZE
    board[coord_to_index(0, 0, 0)] = 1
    board[coord_to_index(4, 4, 2)] = -1
    black_pts, white_pts = _score(jnp.int32(board))
    assert int(black_pts) == 1
    assert int(white_pts) == 1


def test_reward_black_win():
    # Black controls whole board, terminate via two passes from a crafted position.
    board = [0] * BOARD_SIZE
    board[coord_to_index(2, 2, 1)] = 1
    state = _empty_state(board, color=0)
    state = step(state, jnp.int32(PASS_ACTION))
    state = step(state, jnp.int32(PASS_ACTION))
    assert bool(state.terminated)
    # player_order identity -> player 0 is black
    assert state.rewards[0] == 1.0
    assert state.rewards[1] == -1.0
    assert float(state.rewards.sum()) == 0.0


def test_reward_white_win_by_komi():
    # Equal stones, no territory -> white wins by komi.
    board = [0] * BOARD_SIZE
    board[coord_to_index(0, 0, 0)] = 1
    board[coord_to_index(4, 4, 2)] = -1
    black_pts, white_pts = _score(jnp.int32(board))
    assert int(black_pts) == int(white_pts)  # 1 == 1, komi breaks the tie
    state = _empty_state(board, color=0)
    state = step(state, jnp.int32(PASS_ACTION))
    state = step(state, jnp.int32(PASS_ACTION))
    assert bool(state.terminated)
    assert state.rewards[1] == 1.0  # white (player 1) wins
    assert state.rewards[0] == -1.0
    assert float(state.rewards.sum()) == 0.0


def test_rewards_zero_sum_random_play():
    state = init(jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(7)
    for _ in range(200):
        if bool(state.terminated):
            break
        key, sub = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        action = jax.random.categorical(sub, logits)
        state = step(state, action)
    assert float(state.rewards.sum()) == 0.0


# --------------------------------------------------------------------------- #
# Observation tests
# --------------------------------------------------------------------------- #
def test_observation_shape():
    state = init(jax.random.PRNGKey(0))
    assert state.observation.shape == (DEPTH, HEIGHT, WIDTH, 2)


def test_observation_perspective_swaps():
    state = init(jax.random.PRNGKey(0))
    a = coord_to_index(1, 2, 1)  # x=1, y=2, z=1
    state = step(state, jnp.int32(a))  # black plays; now white to move
    cur = int(state.current_player)
    opp = 1 - cur
    obs_current = observe(state, jnp.int32(cur))
    obs_opponent = observe(state, jnp.int32(opp))
    # From the opponent's (black's) perspective, the stone is "mine" (channel 0).
    assert bool(obs_opponent[1, 2, 1, 0])
    assert not bool(obs_opponent[1, 2, 1, 1])
    # From the current player's (white's) perspective, the stone is the opponent's (channel 1).
    assert bool(obs_current[1, 2, 1, 1])
    assert not bool(obs_current[1, 2, 1, 0])


# --------------------------------------------------------------------------- #
# API / registration / JIT / vmap tests
# --------------------------------------------------------------------------- #
def test_api():
    env2 = pgx.make("layer_go")
    pgx.api_test(env2, 3, use_key=False)
    pgx.api_test(env2, 3, use_key=True)


def test_make_and_metadata():
    env2 = pgx.make("layer_go")
    assert env2.id == "layer_go"
    assert env2.num_players == 2
    assert env2.num_actions == ACTION_SIZE
    assert env2.observation_shape == (DEPTH, HEIGHT, WIDTH, 2)
    assert "layer_go" in pgx.available_envs()


def test_random_play_does_not_crash():
    state = init(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    steps = 0
    while not bool(state.terminated) and steps < 500:
        key, sub = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        action = jax.random.categorical(sub, logits)
        state = step(state, action)
        steps += 1
    assert bool(state.terminated)


def test_vmap_batch():
    batch = 8
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    vinit = jax.jit(jax.vmap(env.init))
    vstep = jax.jit(jax.vmap(env.step))
    state = vinit(keys)
    assert state._x.board.shape == (batch, BOARD_SIZE)
    key = jax.random.PRNGKey(2)
    for _ in range(20):
        key, sub = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        actions = jax.vmap(lambda lg, k: jax.random.categorical(k, lg))(logits, jax.random.split(sub, batch))
        state = vstep(state, actions)
    assert state.rewards.shape == (batch, 2)


def test_illegal_action_terminates_with_penalty():
    state = init(jax.random.PRNGKey(0))
    a = coord_to_index(2, 2, 1)
    state = step(state, jnp.int32(a))
    loser = int(state.current_player)
    # play the occupied point -> illegal
    state2 = step(state, jnp.int32(a))
    assert bool(state2.terminated)
    assert state2.rewards[loser] == -1.0


# --------------------------------------------------------------------------- #
# Positional superko / ko
# --------------------------------------------------------------------------- #
def _ko_position():
    """A minimal ko in Layer Go.

    White stone W=(2,1,0) is in atari with its single liberty at the ko point K=(1,1,0);
    the surrounding stones make black's capture at K self-atari with its only liberty back
    at W, so a white recapture at W would recreate the exact starting board (board B0).
    The z-neighbours (..,1) are filled to remove the upward liberties that would otherwise
    break the ko. Returns (B0, K, W).
    """
    b = [0] * BOARD_SIZE
    W = coord_to_index(2, 1, 0)
    K = coord_to_index(1, 1, 0)
    for p in [W, coord_to_index(0, 1, 0), coord_to_index(1, 0, 0), coord_to_index(1, 2, 0), coord_to_index(1, 1, 1)]:
        b[p] = -1  # white
    for p in [coord_to_index(3, 1, 0), coord_to_index(2, 0, 0), coord_to_index(2, 2, 0), coord_to_index(2, 1, 1)]:
        b[p] = 1  # black
    return b, K, W


def test_initial_empty_board_in_history():
    # The empty board's hash is recorded at init, so _is_superko flags it.
    state = init(jax.random.PRNGKey(0))
    empty = jnp.zeros(BOARD_SIZE, dtype=jnp.int8)
    assert bool(_is_superko(state._x.hash_history, state._x.num_history, _board_hash(empty)))
    # A board not in history is not flagged.
    other = jnp.zeros(BOARD_SIZE, dtype=jnp.int8).at[0].set(1)
    assert not bool(_is_superko(state._x.hash_history, state._x.num_history, _board_hash(other)))


def test_immediate_ko_recapture_illegal():
    B0, K, W = _ko_position()
    # Seed history so B0 is a previously-seen board, then let black capture at K.
    state = _empty_state(B0, color=0, history_boards=[B0])
    assert bool(state.legal_action_mask[K])  # black capture is legal
    state = step(state, jnp.int32(K))
    assert state._x.board[W] == 0  # white stone captured
    assert state._x.board[K] == 1  # black stone placed
    assert not bool(state.terminated)
    # White recapture at W would recreate B0 -> illegal by superko, in the mask...
    assert not bool(state.legal_action_mask[W])
    # ...and enforced by step (illegal action -> acting player loses).
    loser = int(state.current_player)
    after = step(state, jnp.int32(W))
    assert bool(after.terminated)
    assert after.rewards[loser] == -1.0


def test_non_repeating_capture_is_legal():
    # Same ko shape, but instead of the repeating recapture white plays elsewhere: legal.
    B0, K, W = _ko_position()
    state = _empty_state(B0, color=0, history_boards=[B0])
    state = step(state, jnp.int32(K))
    elsewhere = coord_to_index(4, 4, 2)
    assert bool(state.legal_action_mask[elsewhere])
    state2 = step(state, jnp.int32(elsewhere))
    assert not bool(state2.terminated)
    assert state2._x.board[elsewhere] == -1  # white played, game continues


def test_superko_helper_detects_longer_cycle():
    # Helper-level test for a repetition longer than simple ko: a candidate equal to an
    # older (non-immediately-preceding) recorded board is a superko violation.
    b1 = [0] * BOARD_SIZE
    b1[coord_to_index(0, 0, 0)] = 1
    b2 = [0] * BOARD_SIZE
    b2[coord_to_index(0, 0, 0)] = 1
    b2[coord_to_index(4, 4, 2)] = -1
    b3 = [0] * BOARD_SIZE
    b3[coord_to_index(2, 2, 1)] = 1
    hist, num = _history_from_boards([[0] * BOARD_SIZE, b1, b2, b3])
    assert bool(_is_superko(hist, num, _board_hash(jnp.int8(b1))))  # recreating b1 (older board) is a violation
    assert bool(_is_superko(hist, num, _board_hash(jnp.int8(b2))))
    fresh = [0] * BOARD_SIZE
    fresh[coord_to_index(1, 1, 1)] = -1
    assert not bool(_is_superko(hist, num, _board_hash(jnp.int8(fresh))))  # never-seen board is fine


def test_pass_not_blocked_by_superko_and_terminates():
    # Even with a board that is in history, pass stays legal and two passes terminate.
    B0, K, W = _ko_position()
    state = _empty_state(B0, color=0, history_boards=[B0])
    state = step(state, jnp.int32(K))  # black captures
    assert bool(state.legal_action_mask[PASS_ACTION])  # white can still pass
    state = step(state, jnp.int32(PASS_ACTION))
    assert not bool(state.terminated)
    assert bool(state.legal_action_mask[PASS_ACTION])
    state = step(state, jnp.int32(PASS_ACTION))
    assert bool(state.terminated)  # two consecutive passes


# --------------------------------------------------------------------------- #
# History bookkeeping
# --------------------------------------------------------------------------- #
def test_history_shape_and_init():
    state = init(jax.random.PRNGKey(0))
    assert state._x.hash_history.shape == (MAX_HISTORY, 2)
    assert int(state._x.num_history) == 1  # only the empty board


def test_stone_move_records_history_pass_does_not():
    state = init(jax.random.PRNGKey(0))
    state = step(state, jnp.int32(coord_to_index(1, 1, 0)))
    assert int(state._x.num_history) == 2  # empty + one stone board
    assert int(state._x.step_count) == 1
    # the recorded hash matches the current board's hash
    assert bool((state._x.hash_history[1] == _board_hash(state._x.board)).all())
    state = step(state, jnp.int32(PASS_ACTION))
    assert int(state._x.num_history) == 2  # pass adds no row
    assert int(state._x.step_count) == 2  # but the ply counter advances


def test_unwritten_history_rows_excluded_by_num_history():
    # Validity is tracked by num_history: rows at or beyond it are never compared, even if
    # they happen to hold a real board's hash.
    planted = [0] * BOARD_SIZE
    planted[coord_to_index(3, 1, 2)] = -1
    hist, num = _history_from_boards([[0] * BOARD_SIZE, [0] * BOARD_SIZE])  # num_history = 2
    hist = hist.at[5].set(_board_hash(jnp.int8(planted)))  # plant a hash at row 5 (>= num_history)
    assert not bool(_is_superko(hist, num, _board_hash(jnp.int8(planted))))  # masked out by num_history=2
    assert bool(_is_superko(hist, jnp.int32(6), _board_hash(jnp.int8(planted))))  # visible once num_history>5


# --------------------------------------------------------------------------- #
# Finite-episode termination (max game length)
# --------------------------------------------------------------------------- #
def test_max_game_length_terminates_and_area_scores():
    # One ply before the cap, a single legal stone move hits MAX_GAME_LENGTH and terminates.
    board = [0] * BOARD_SIZE  # empty board at the cap boundary
    state = _empty_state(board, player_order=(0, 1), step_count=MAX_GAME_LENGTH - 1)
    assert not bool(state.terminated)
    mover = int(state.current_player)  # this player's lone stone will control the whole board
    a = coord_to_index(2, 2, 1)
    state = step(state, jnp.int32(a))
    assert int(state._x.step_count) == MAX_GAME_LENGTH
    assert bool(state.terminated)  # max-length termination
    # area scored: the single placed stone controls the whole board -> that player wins, zero-sum
    assert float(state.rewards[mover]) == 1.0
    assert float(state.rewards[1 - mover]) == -1.0
    assert float(state.rewards.sum()) == 0.0


def test_one_ply_before_cap_does_not_terminate():
    board = [0] * BOARD_SIZE
    state = _empty_state(board, player_order=(0, 1), step_count=MAX_GAME_LENGTH - 2)
    state = step(state, jnp.int32(coord_to_index(0, 0, 0)))
    assert int(state._x.step_count) == MAX_GAME_LENGTH - 1
    assert not bool(state.terminated)


def test_illegal_before_cap_uses_illegal_loss_not_area():
    # An illegal action just below the cap follows illegal-action behavior (acting player
    # loses with -1), not area scoring of the board.
    board = [0] * BOARD_SIZE
    board[coord_to_index(2, 2, 1)] = 1  # occupy a point
    state = _empty_state(board, player_order=(0, 1), step_count=MAX_GAME_LENGTH - 1)
    loser = int(state.current_player)
    state = step(state, jnp.int32(coord_to_index(2, 2, 1)))  # occupied -> illegal
    assert bool(state.terminated)
    assert float(state.rewards[loser]) == -1.0
    assert float(state.rewards.sum()) == 0.0


# --------------------------------------------------------------------------- #
# Pass / two-pass termination
# --------------------------------------------------------------------------- #
def test_pass_counter_resets_after_stone_move():
    state = init(jax.random.PRNGKey(0))
    state = step(state, jnp.int32(PASS_ACTION))
    assert int(state._x.consecutive_pass_count) == 1
    state = step(state, jnp.int32(coord_to_index(2, 2, 1)))  # stone move resets the counter
    assert int(state._x.consecutive_pass_count) == 0
    assert not bool(state.terminated)
    state = step(state, jnp.int32(PASS_ACTION))
    assert int(state._x.consecutive_pass_count) == 1
    assert not bool(state.terminated)
    state = step(state, jnp.int32(PASS_ACTION))
    assert int(state._x.consecutive_pass_count) == 2
    assert bool(state.terminated)


# --------------------------------------------------------------------------- #
# Board-wide legal mask: differential test vs the straightforward definition
# --------------------------------------------------------------------------- #
def _legal_action_mask_reference(board, color, hash_history, num_history):
    """Reference legal mask: the straightforward per-candidate definition (empty, not suicide
    after captures, not positional-superko), used to pin the optimized board-wide mask."""
    my_sign = _color_to_sign(color)

    def is_legal(a):
        is_empty = board[a] == 0
        new_board, _ = _resulting_board(board, my_sign, a)
        not_suicide = _group_alive(new_board, new_board == my_sign)[a]
        not_superko = ~_is_superko(hash_history, num_history, _board_hash(new_board))
        return is_empty & not_suicide & not_superko

    return jnp.append(jax.vmap(is_legal)(jnp.arange(BOARD_SIZE)), True)


def test_legal_action_mask_matches_reference():
    # Lockstep random-legal playouts (batch 64): the optimized board-wide mask must equal the
    # per-candidate reference at every ply, covering captures, suicide, atari and superko.
    batch = 64
    vinit = jax.jit(jax.vmap(env.init))
    vstep = jax.jit(jax.vmap(env.step))
    ref = jax.jit(jax.vmap(_legal_action_mask_reference, in_axes=(0, 0, 0, 0)))
    fast = jax.jit(jax.vmap(_legal_action_mask, in_axes=(0, 0, 0, 0)))

    state = vinit(jax.random.split(jax.random.PRNGKey(20240611), batch))
    key = jax.random.PRNGKey(7)
    checked = 0
    for _ in range(160):
        x = state._x
        assert bool(
            (
                ref(x.board, x.color, x.hash_history, x.num_history)
                == fast(x.board, x.color, x.hash_history, x.num_history)
            ).all()
        ), "board-wide legal mask diverged from reference"
        checked += 1
        if bool(state.terminated.all()):
            break
        key, sub = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        actions = jax.vmap(lambda lg, k: jax.random.categorical(k, lg))(logits, jax.random.split(sub, batch))
        state = vstep(state, actions)
    assert checked > 50  # sanity: the playout actually ran
