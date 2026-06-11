import jax
import jax.numpy as jnp

import pgx
from pgx.layer_go import (
    ACTION_SIZE,
    BOARD_SIZE,
    DEPTH,
    HEIGHT,
    KOMI,
    NEIGHBORS,
    PASS_ACTION,
    WIDTH,
    GameState,
    LayerGo,
    State,
    _legal_action_mask,
    _score,
    coord_to_index,
    index_to_coord,
)

env = LayerGo()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def _empty_state(board, color=0, player_order=(0, 1)):
    """Build a State with a hand-crafted board and the given color to move."""
    x = GameState(step_count=jnp.int32(color), board=jnp.int32(board))
    return State(  # type: ignore
        current_player=jnp.int32(player_order[color]),
        legal_action_mask=_legal_action_mask(jnp.int32(board), jnp.int32(color)),
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
