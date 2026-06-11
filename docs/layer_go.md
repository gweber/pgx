# Layer Go

Layer Go is a compact 3D Go variant played on a **5 × 5 × 3** orthogonal grid (75 points).
It keeps the rules of Go — connection, liberties, captures, suicide, passing, area scoring —
but plays them on a stack of three 5×5 layers connected along the z-axis.

## Usage

```py
import pgx

env = pgx.make("layer_go")
```

or you can directly load the `LayerGo` class

```py
from pgx.layer_go import LayerGo

env = LayerGo()
```

A minimal episode:

```py
import jax
import pgx

env = pgx.make("layer_go")
state = env.init(jax.random.PRNGKey(0))
state = env.step(state, 0)   # place a stone at point index 0 == (x=0, y=0, z=0)
state = env.step(state, 75)  # pass (action 75)
```

## Board and indexing

Points are addressed by coordinates `(x, y, z)`:

- `x ∈ [0, 4]` (width)
- `y ∈ [0, 4]` (height)
- `z ∈ [0, 2]` (depth / layer)

The linear index of a point is

```python
index = z * WIDTH * HEIGHT + y * WIDTH + x        # WIDTH = HEIGHT = 5
```

and the reverse mapping is

```python
z = index // (WIDTH * HEIGHT)
rem = index % (WIDTH * HEIGHT)
y = rem // WIDTH
x = rem % WIDTH
```

Examples: `(0,0,0) → 0`, `(4,0,0) → 4`, `(0,1,0) → 5`, `(0,0,1) → 25`, `(4,4,2) → 74`.

## Adjacency

Adjacency is **orthogonal only**. A point has up to 6 neighbours:

```
(x-1, y, z) (x+1, y, z) (x, y-1, z) (x, y+1, z) (x, y, z-1) (x, y, z+1)
```

There is **no diagonal adjacency**. Diagonal touching — within a layer or across layers —
does not connect stones into a group and does not provide a liberty. Neighbour counts:
corners have 3 neighbours, outer-layer face centres have 5, and a fully interior point such
as `(2,2,1)` has 6.

## Groups, liberties, and captures

- Same-colour stones connected through orthogonal adjacency form a **group**.
- A **liberty** is an empty point orthogonally adjacent to any stone of the group.
- When a stone is placed:
  1. adjacent opponent groups are examined,
  2. any opponent group left with **zero liberties** is removed (captured),
  3. the placed stone's own group is then examined.

Captures work across layers exactly like in-plane captures: a group whose only remaining
liberty is the point directly above or below it is captured when that point is filled.

## Suicide

A move is **suicide** (and therefore illegal) if, *after* resolving opponent captures, the
placed stone's own group has zero liberties. A move that removes its own last liberty but
captures an adjacent opponent group — thereby opening a liberty — is legal.

## Ko and repetition

Layer Go uses **positional superko**: a stone placement is illegal if, after captures,
the resulting board position is equal to any previous board position in the same game.
This prevents immediate ko recapture and longer repetition cycles.

Only the stone arrangement on the 75-point board is compared — never the player to move
(this is *positional*, not *situational*, superko). Passes do not change the board position
and are always legal in non-terminal states. The initial empty board is part of the superko
history, and every board reached by a legal stone move is added to it; passes add no entry.

## Termination

A game ends when one of the following happens:

1. **two consecutive passes**,
2. the fixed **maximum game length** (`512` plies) is reached,
3. an **illegal action** is taken.

Normal games are scored by area scoring. If the maximum game length is reached, the current
board is also scored by area scoring (zero-sum, winner `+1` / loser `-1`). Illegal actions
follow standard PGX behavior: the acting player loses immediately (and the board is *not*
area-scored). There is no separate "board full" rule — when no point move is legal both
players simply pass, and the two-pass rule terminates the game cleanly.

## Scoring (area scoring + komi)

```
black_score = black_stones_on_board + black_controlled_empty_points
white_score = white_stones_on_board + white_controlled_empty_points + komi   # komi = 7.5
```

Empty territory is assigned by flood-filling connected empty regions with orthogonal
adjacency:

- a region belongs to **black** if every bordering stone is black and at least one bordering
  stone exists,
- a region belongs to **white** under the symmetric condition,
- a region bordered by both colours, or bordered by no stones, is **neutral**.

Black wins if `black_score > white_score`, otherwise white wins. Because komi is fractional
(7.5) there are no draws.

## Specs

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `76 (= 5 × 5 × 3 + 1)` |
| Observation shape | `(3, 5, 5, 2)` = `(depth, height, width, channels)` |
| Observation type | `bool` |
| Rewards | `{-1, 1}` at terminal, `0` otherwise |
| Komi | `7.5` |
| Maximum game length | `512` plies |

## Observation

The observation is current-player-relative with shape `(DEPTH, HEIGHT, WIDTH, 2)`:

| Index | Description |
|:---:|:----|
| `[:, :, :, 0]` | the observing player's stones |
| `[:, :, :, 1]` | the opponent's stones |

The first axis is the layer `z`, the second is `y`, the third is `x`, matching the linear
index formula above.

## Action

Actions `0 … 74` place a stone on the corresponding point index. Action `75` is pass.
A point action is legal iff (1) the game is not terminal, (2) the point is empty, (3) the
placement is not suicide after captures, and (4) the resulting board has not appeared before
in the game (positional superko). Pass is legal whenever the game is not terminal and is never
subject to the superko check. The legal action mask and `step` agree: a superko-violating
point move is `False` in the mask, and selecting it anyway triggers standard illegal-action
termination.

## Rewards

Standard PGX zero-sum terminal reward: the winner receives `+1`, the loser `-1`, and all
non-terminal steps give `0`. As with the other PGX games, taking an illegal action ends the
game immediately with `-1` for the offending player.

## Known limitations

- **Fixed board size.** Only the 5×5×3 board is implemented.
- **Fixed history size.** Superko history is bounded by the maximum game length (`512` plies),
  stored as a fixed `(513, 75)` int8 buffer in the state (~38 KB/state).
- **Komi is untuned.** `komi = 7.5` is a starting value and has not been balanced for this 3D
  board; it serves to break ties.
- **No variable-size engine.** Larger Layer Go variants such as 7×7×3 or 5×5×5 are not
  implemented yet.

## Random selfplay sanity check

`examples/layer_go_random_selfplay.py` runs many episodes of uniformly-random **legal**
self-play (vmapped + jitted) and prints first-order health metrics: average/min/max game
length, black/white win rates, average final score margin, terminal-reason distribution
(two passes / max length / illegal), average legal actions per ply, percentage of games
reaching the max length, and a rough env-steps/sec throughput.

```sh
python examples/layer_go_random_selfplay.py --episodes 64 --seed 0
```

It only samples from `legal_action_mask` and reads public state (plus the internal
`_score` helper for the area-score margin), so it changes no game rules. Terminal reasons are
*inferred* from public state — `consecutive_pass_count >= 2` for the two-pass ending and
`step_count >= 512` for the max-length ending — because the environment does not expose a
terminal-reason field. The legal-only sampler should never produce the "illegal action"
bucket (it stays at 0).

**This is a sanity check, not a balance study.** `komi = 7.5` is untuned, and a lopsided
random-play win rate reflects the komi value and random dynamics on this small 3D board, not
the strength balance of skilled play. Throughput is also CPU-bound in this slice; use a
smaller `--episodes` for a quick check or an accelerator for larger runs.

## Visualization

Layer Go supports the standard PGX SVG visualizer via `state.to_svg()` / `state.save_svg(...)`
(and inline rendering in notebooks). The three z-layers are drawn as separate 5×5 Go-style
boards laid out left-to-right and labeled `z=0`, `z=1`, `z=2`; faint dotted connectors across
the gaps indicate the vertical (z-axis) adjacency between layers.
