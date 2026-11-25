#!/usr/bin/env python3
"""Grow manganese-dendrite-like patterns by diffusion-limited aggregation.

This is a toy. Real fissure dendrites grow in a film of micropores on a
joint or bedding plane, with metal supplied by diffusion; van Straaten
(1978) argued the branched shape wins because the average travel
distance for ions is shorter than for a solid blob. DLA is the standard
simple model of that kind of growth, not a mineralogy simulation.
Units are pixels. Nothing here is a timescale.

Griff, 24 Nov 2026, own time.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dla_from_seeds(
    width: int,
    height: int,
    seeds: list[tuple[int, int]],
    n_particles: int,
    rng: random.Random,
    launch_margin: int = 12,
    kill_extra: int = 40,
    stick_diag: bool = True,
) -> np.ndarray:
    """On-lattice DLA. Returns occupied boolean array (height, width)."""
    occupied = np.zeros((height, width), dtype=np.bool_)
    for x, y in seeds:
        if 0 <= x < width and 0 <= y < height:
            occupied[y, x] = True

    # bounding box of the cluster, expanded as it grows
    ys, xs = np.nonzero(occupied)
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())

    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if stick_diag:
        neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Precompute: a walker sticks if any neighbour is occupied.
    # We keep a slightly expanded "alive" window.
    added = 0
    attempts = 0
    max_attempts = n_particles * 40

    def in_kill(x, y) -> bool:
        # kill if far outside the current bounding box
        return (
            x < minx - kill_extra
            or x > maxx + kill_extra
            or y < miny - kill_extra
            or y > maxy + kill_extra
            or x < 0
            or y < 0
            or x >= width
            or y >= height
        )

    def launch_point() -> tuple[int, int]:
        # Launch on a rectangle a fixed margin outside the bounding box,
        # clipped to the grid.
        x0 = max(0, minx - launch_margin)
        x1 = min(width - 1, maxx + launch_margin)
        y0 = max(0, miny - launch_margin)
        y1 = min(height - 1, maxy + launch_margin)
        # pick a point on the perimeter of that rectangle
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        peri = 2 * (w + h)
        t = rng.randrange(peri)
        if t < w:
            return x0 + t, y0
        t -= w
        if t < h:
            return x1, y0 + t
        t -= h
        if t < w:
            return x1 - t, y1
        t -= w
        return x0, y1 - t

    def is_stuck(x, y) -> bool:
        for dx, dy in neigh:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and occupied[ny, nx]:
                return True
        return False

    while added < n_particles and attempts < max_attempts:
        attempts += 1
        x, y = launch_point()
        if occupied[y, x]:
            continue
        steps = 0
        max_steps = (kill_extra + launch_margin) * 80
        stuck = False
        while steps < max_steps:
            steps += 1
            # random 4-neighbour step
            d = rng.randrange(4)
            if d == 0:
                x -= 1
            elif d == 1:
                x += 1
            elif d == 2:
                y -= 1
            else:
                y += 1
            if in_kill(x, y):
                break
            if is_stuck(x, y):
                occupied[y, x] = True
                if x < minx:
                    minx = x
                if x > maxx:
                    maxx = x
                if y < miny:
                    miny = y
                if y > maxy:
                    maxy = y
                added += 1
                stuck = True
                break
        # if not stuck, just launch another
        _ = stuck

    return occupied


def thicken(occupied: np.ndarray, passes: int = 1) -> np.ndarray:
    """Van Straaten: later supply thickens stems and branches without
    growing the overall size much. A few dilate-on-existing-neighbour
    passes, biased toward already-thick parts (more occupied neighbours).
    """
    out = occupied.copy()
    h, w = out.shape
    for _ in range(passes):
        # count occupied 8-neighbours
        pad = np.pad(out.astype(np.uint8), 1)
        n = (
            pad[0:-2, 0:-2]
            + pad[0:-2, 1:-1]
            + pad[0:-2, 2:]
            + pad[1:-1, 0:-2]
            + pad[1:-1, 2:]
            + pad[2:, 0:-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        )
        # grow into empty cells that already touch the cluster,
        # more readily if they have several occupied neighbours
        edge = (~out) & (n >= 2)
        # keep a fraction so it doesn't fill in the gaps
        grow = edge & (n >= 3)
        # also a thinner chance for n==2
        grow = grow | (edge & (n == 2) & (np.random.random(out.shape) < 0.25))
        out = out | grow
    return out


def limestone_base(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Cream Solnhofen-ish limestone, slight grain, a few darker flecks."""
    base = np.zeros((height, width, 3), dtype=np.float32)
    # warm cream
    base[..., 0] = 214
    base[..., 1] = 204
    base[..., 2] = 178
    # low-frequency blotch
    yy, xx = np.mgrid[0:height, 0:width]
    blotch = (
        0.55 * np.sin(xx / 73.0 + yy / 91.0)
        + 0.30 * np.sin(xx / 31.0 - yy / 47.0)
        + 0.15 * np.sin((xx + yy) / 19.0)
    )
    blotch = (blotch - blotch.min()) / (blotch.max() - blotch.min() + 1e-6)
    for c, amp in ((0, 16), (1, 14), (2, 18)):
        base[..., c] += (blotch - 0.5) * amp
    # fine grain
    grain = rng.normal(0, 4.5, (height, width))
    base += grain[..., None]
    # rare darker grains (foraminifera ghosts, nothing labelled)
    n_flecks = int(width * height * 0.0008)
    fy = rng.integers(0, height, n_flecks)
    fx = rng.integers(0, width, n_flecks)
    for y, x in zip(fy, fx):
        r = int(rng.integers(1, 3))
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        base[y0:y1, x0:x1] *= rng.uniform(0.85, 0.95)
    return np.clip(base, 0, 255)


def paint_dendrites(occupied: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Black-brown oxide on cream stone. Slight softness so it isn't pixels."""
    h, w = occupied.shape
    # distance-ish thickness already in the mask; darken more at cores
    pad = np.pad(occupied.astype(np.uint8), 1)
    n = (
        pad[0:-2, 0:-2]
        + pad[0:-2, 1:-1]
        + pad[0:-2, 2:]
        + pad[1:-1, 0:-2]
        + pad[1:-1, 2:]
        + pad[2:, 0:-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    )
    core = occupied & (n >= 4)
    mid = occupied & ~core
    img = base.copy()
    # manganese oxides: near-black with a faint brown
    img[mid] = img[mid] * 0.12 + np.array([28, 22, 16], dtype=np.float32)
    img[core] = img[core] * 0.04 + np.array([18, 14, 10], dtype=np.float32)
    # a one-pixel soft fringe, slightly rusty
    fringe = (~occupied) & (n >= 1)
    img[fringe] = img[fringe] * 0.72 + np.array([50, 38, 28], dtype=np.float32) * 0.28
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_crack(base: np.ndarray, x0: int, y0: int, x1: int, y1: int, rng: np.random.Generator) -> np.ndarray:
    """A thin irregular joint, slightly darker than the stone."""
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    # walk a jittered line
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t + float(rng.normal(0, 0.8))
        y = y0 + (y1 - y0) * t + float(rng.normal(0, 0.8))
        pts.append((x, y))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.line(pts, fill=(70, 55, 40, 140), width=2)
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return np.array(out.convert("RGB"), dtype=np.float32)


def add_caption(img: Image.Image, lines: list[str]) -> Image.Image:
    """White caption bar under the stone."""
    bar_h = 28 + 18 * len(lines)
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (245, 243, 238))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(13)
    y = h + 8
    for i, line in enumerate(lines):
        d.text((12, y), line, fill=(40, 36, 32), font=font if i == 0 else small)
        y += 18 if i == 0 else 16
    return canvas


def run_tree(rng_seed: int = 1896) -> Path:
    """Tree-shaped dendrites growing away from a joint, as in Solnhofen."""
    rng = random.Random(rng_seed)
    npr = np.random.default_rng(rng_seed)
    w, h = 900, 620
    # joint along the left edge, a little in
    joint_x = 48
    seeds = [(joint_x + rng.randint(0, 1), y) for y in range(20, h - 20, 3)]
    # also a few seeds slightly off the joint (specks)
    for _ in range(8):
        seeds.append((joint_x + rng.randint(0, 2), rng.randint(30, h - 30)))
    occ = dla_from_seeds(w, h, seeds, n_particles=18000, rng=rng, launch_margin=14, kill_extra=50)
    occ = thicken(occ, passes=1)
    # don't thicken over the far empty field much; already handled
    base = limestone_base(w, h, npr)
    base = draw_crack(base, joint_x, 8, joint_x, h - 8, npr)
    painted = paint_dendrites(occ, base)
    img = Image.fromarray(painted).filter(ImageFilter.GaussianBlur(radius=0.45))
    # re-sharpen a touch so branches stay crisp
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    captioned = add_caption(
        img,
        [
            "Tree-shaped dendrites growing away from a joint. Toy DLA, not a photograph.",
            "Manganese oxides in stone look like this. They are not ferns. Griff, 24 Nov 2026.",
        ],
    )
    path = OUT / "dendrites-tree.png"
    captioned.save(path, "PNG")
    return path


def run_stars(rng_seed: int = 1978) -> Path:
    """Star-shaped dendrites from isolated nuclei."""
    rng = random.Random(rng_seed)
    npr = np.random.default_rng(rng_seed)
    w, h = 900, 620
    # a few nuclei scattered on the bedding plane
    seeds = []
    centres = [(220, 180), (510, 300), (720, 160), (400, 480), (150, 430), (780, 470)]
    for cx, cy in centres:
        seeds.append((cx, cy))
        for _ in range(4):
            seeds.append((cx + rng.randint(-1, 1), cy + rng.randint(-1, 1)))
    occ = dla_from_seeds(w, h, seeds, n_particles=14000, rng=rng, launch_margin=12, kill_extra=45)
    occ = thicken(occ, passes=1)
    base = limestone_base(w, h, npr)
    painted = paint_dendrites(occ, base)
    img = Image.fromarray(painted).filter(ImageFilter.GaussianBlur(radius=0.45))
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    captioned = add_caption(
        img,
        [
            "Star-shaped dendrites from isolated nuclei. Toy DLA, not a photograph.",
            "Same chemistry as the joint trees; only the starting points changed. Griff, 24 Nov 2026.",
        ],
    )
    path = OUT / "dendrites-stars.png"
    captioned.save(path, "PNG")
    return path


if __name__ == "__main__":
    print("growing trees...", flush=True)
    p1 = run_tree()
    print("wrote", p1, flush=True)
    print("growing stars...", flush=True)
    p2 = run_stars()
    print("wrote", p2, flush=True)
