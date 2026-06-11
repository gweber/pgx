import jax.numpy as jnp

from pgx.layer_go import DEPTH, HEIGHT, WIDTH, State


def _make_layer_go_dwg(dwg, state: State, config):
    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]

    # Each z-layer is drawn as its own 5x5 Go-style board. The layers are placed
    # left-to-right (z = 0, 1, 2) with a gap between them; faint dotted connectors
    # across each gap hint at the vertical (z-axis) adjacency between layers.
    layer_span = (WIDTH - 1) * GRID_SIZE  # grid extent of one layer
    gap = 2 * GRID_SIZE
    layer_pitch = layer_span + gap  # x distance between consecutive layer origins

    board = jnp.clip(state._x.board, -1, 1)

    board_g = dwg.g()

    # z-axis connectors (drawn first so stones sit on top)
    for z in range(DEPTH - 1):
        x_start = z * layer_pitch + layer_span
        x_end = (z + 1) * layer_pitch
        for y in range(HEIGHT):
            board_g.add(
                dwg.line(
                    start=(x_start, y * GRID_SIZE),
                    end=(x_end, y * GRID_SIZE),
                    stroke=color_set.grid_color,
                    stroke_width="0.5px",
                    stroke_dasharray="2,3",
                )
            )

    for z in range(DEPTH):
        ox = z * layer_pitch
        layer_g = dwg.g()

        # grid lines
        hlines = layer_g.add(dwg.g(id=f"hlines_{z}", stroke=color_set.grid_color))
        for y in range(1, HEIGHT - 1):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(GRID_SIZE * (WIDTH - 1), GRID_SIZE * y),
                    stroke_width="0.5px",
                )
            )
        vlines = layer_g.add(dwg.g(id=f"vlines_{z}", stroke=color_set.grid_color))
        for x in range(1, WIDTH - 1):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(GRID_SIZE * x, GRID_SIZE * (HEIGHT - 1)),
                    stroke_width="0.5px",
                )
            )
        layer_g.add(
            dwg.rect(
                (0, 0),
                ((WIDTH - 1) * GRID_SIZE, (HEIGHT - 1) * GRID_SIZE),
                fill="none",
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )
        # center point (hoshi)
        layer_g.add(
            dwg.circle(
                center=(2 * GRID_SIZE, 2 * GRID_SIZE),
                r=GRID_SIZE / 10,
                fill=color_set.grid_color,
            )
        )

        # stones
        for y in range(HEIGHT):
            for x in range(WIDTH):
                stone = int(board[z * WIDTH * HEIGHT + y * WIDTH + x])
                if stone == 0:
                    continue
                color = color_set.p1_color if stone == 1 else color_set.p2_color
                outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
                layer_g.add(
                    dwg.circle(
                        center=(x * GRID_SIZE, y * GRID_SIZE),
                        r=GRID_SIZE / 2.2,
                        stroke=outline,
                        fill=color,
                    )
                )

        layer_g.translate(ox, 0)
        board_g.add(layer_g)

        # z-layer label below the board
        board_g.add(
            dwg.text(
                text=f"z={z}",
                insert=(ox + layer_span / 2, (HEIGHT - 1) * GRID_SIZE + GRID_SIZE),
                fill=color_set.text_color,
                font_size=f"{int(GRID_SIZE * 0.7)}px",
                font_family="Courier",
                text_anchor="middle",
            )
        )

    board_g.translate(GRID_SIZE / 2, GRID_SIZE / 2)
    return board_g
