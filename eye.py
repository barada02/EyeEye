"""EyeEye - a living eye drawn entirely with text characters.

Controls
  mouse         look around (the eye follows the cursor)
  hold click    shine a light - the pupil constricts
  space         dive into the pupil / come back out
  B             blink
  + / -         bigger / smaller characters
  G             toggle background glow
  S             save a screenshot
  H             show / hide help
  F11           fullscreen
  Esc           quit
"""

import ctypes
import math
import sys
import time

import numpy as np
import pygame

LEVELS = 20      # brightness steps in each character ramp
VARIANTS = 6     # interchangeable characters per step (cells drift between them)

# Every region of the face is written in its own alphabet.
FAMILIES = (
    "`.',:;-~!ilrjftc1/\\|()?7ivxzsJLYTCZunoe",              # skin
    "`.'-:;~=+*^ocszaxe%&#@",                                # sclera
    "`.':;ilrjcvunxzoaeJLCUYXZOQ0DGBRKHMNWAPSE8%$&#@",      # iris
    "`.',-:;~=oO0@",                                         # space (stars get their own glyphs)
)
SKIN, SCLERA, IRIS, SPACE = range(4)
STROKES = "-/|\\"   # line glyphs for 0, 45, 90, 135 degrees

WO = 1.0          # half-width of the eye opening (world units; screen half-height = 1)
RI = 0.36         # iris radius
RP = 0.13         # resting pupil radius
BALL = 1.25       # eyeball radius
ZOOM0, CAM0 = 1.35, (0.0, 0.10)
ZOOM_IN = 14.0    # camera zoom once we are through the pupil
DIVE_TIME = 2.6   # seconds to fall into the pupil


# ------------------------------------------------------------------ helpers

def ss(a, b, x):
    """Smoothstep; a > b gives the falling version."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a + (b - a) * t[..., None]


def rgb(r, g, b):
    return np.array([r, g, b], np.float32)


def tile_noise(n, octaves, rng):
    """Tileable fractal value noise in [0, 1], shape (n, n)."""
    out = np.zeros((n, n), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 << o
        grid = rng.random((cells, cells), dtype=np.float32)
        x = np.arange(n, dtype=np.float32) * cells / n
        i0 = x.astype(np.int64)
        f = x - i0
        f = f * f * (3 - 2 * f)
        i1 = (i0 + 1) % cells
        rows = grid[i0] * (1 - f)[:, None] + grid[i1] * f[:, None]
        out += amp * (rows[:, i0] * (1 - f) + rows[:, i1] * f)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def bil(T, x, y):
    """Bilinear, wrapping lookup of square power-of-two texture T at texel coords (x, y)."""
    m = T.shape[0] - 1
    flat = T.ravel()
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    x0 = xf.astype(np.int32) & m
    x1 = (x0 + 1) & m
    y0 = (yf.astype(np.int32) & m) * (m + 1)
    y1 = (y0 + m + 1) & (m * (m + 1) + m)
    a, b = flat.take(y0 + x0), flat.take(y0 + x1)
    c, d = flat.take(y1 + x0), flat.take(y1 + x1)
    top = a + (b - a) * fx
    return top + (c + (d - c) * fx - top) * fy


def stroke_index(phi):
    """Line orientation in degrees (world space, y up) -> index into STROKES."""
    return np.rint((phi % 180.0) / 45.0).astype(np.int64) % 4


_rng = np.random.default_rng(7)
TN = tile_noise(256, 6, _rng)                      # smooth fractal noise
TW = _rng.random((256, 256), dtype=np.float32)     # white noise


# ------------------------------------------------------------------ glyphs

class Glyphs:
    """Pre-rendered character atlas plus brightness -> character tables."""

    def __init__(self, size):
        font = pygame.font.SysFont("consolas,cascadiamono,dejavusansmono,menlo,couriernew", size)
        self.cw, self.ch = font.size("M")
        chars = [chr(c) for c in range(32, 127)]
        self.index = {c: i for i, c in enumerate(chars)}
        atlas = np.zeros((len(chars), self.ch, self.cw), np.uint8)
        for i, c in enumerate(chars):
            a = pygame.surfarray.array_red(font.render(c, True, (255, 255, 255), (0, 0, 0))).T
            h, w = min(a.shape[0], self.ch), min(a.shape[1], self.cw)
            atlas[i, :h, :w] = a[:h, :w]
        self.atlas = atlas

        cov = atlas.reshape(len(chars), -1).mean(1) / 255.0
        self.lut = np.zeros((len(FAMILIES), LEVELS, VARIANTS), np.int64)
        for f, charset in enumerate(FAMILIES):
            ids = np.array(sorted({self.index[c] for c in charset}))
            for k in range(LEVELS):
                target = cov.max() * k / (LEVELS - 1)
                self.lut[f, k] = ids[np.argsort(np.abs(cov[ids] - target))][:VARIANTS]
        self.lut[:, 0, :] = self.index[" "]
        self.strokes = np.array([self.index[c] for c in STROKES])
        self.star = np.array([self.index[c] for c in ".+*"])

    def compose(self, idx, fg, bg=None):
        """Character ids (H, W) + uint8 colours (H, W, 3) -> a pygame Surface.

        The glyph mask is built in numpy; tinting is left to pygame's blitters,
        which are far faster than numpy at per-pixel RGB work. Everything stays
        32-bit so the blitters can take their SIMD paths.
        """
        H, W = idx.shape
        size = (W * self.cw, H * self.ch)
        mask = np.ascontiguousarray(self.atlas[idx].transpose(0, 2, 1, 3)).astype(np.uint32)
        mask *= 0x010101                                     # grey RGBX pixels
        out = self._cells_surface(fg, size)
        out.blit(pygame.image.frombuffer(mask, size, "RGBX"), (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        if bg is not None:
            out.blit(self._cells_surface(bg, size), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        return out

    @staticmethod
    def _cells_surface(colours, size):
        """One colour per cell, blown up to one block of pixels per cell."""
        H, W, _ = colours.shape
        rgbx = np.zeros((H, W, 4), np.uint8)
        rgbx[..., :3] = colours
        return pygame.transform.scale(pygame.image.frombuffer(rgbx, (W, H), "RGBX"), size)


# ------------------------------------------------------------------ space

class Stars:
    SPREAD = 8.0

    def __init__(self, n, rng):
        self.rng = rng
        self.p = rng.uniform(-self.SPREAD, self.SPREAD, (n, 2)).astype(np.float32)
        self.z = rng.uniform(0.05, 1.0, n).astype(np.float32)
        self.mag = (rng.uniform(0.3, 1.0, n) ** 2).astype(np.float32)
        self.phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        hue = rng.random(n)[:, None]
        self.col = np.where(hue < 0.25, rgb(0.70, 0.80, 1.00),
                            np.where(hue > 0.85, rgb(1.00, 0.82, 0.60), rgb(1.0, 1.0, 1.0)))

    def step(self, dt, speed):
        self.z -= speed * dt
        dead = self.z < 0.03
        k = int(dead.sum())
        if k:
            self.p[dead] = self.rng.uniform(-self.SPREAD, self.SPREAD, (k, 2))
            self.z[dead] = 1.0


# ------------------------------------------------------------------ the eye

class Eye:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.t = 0.0
        self.gaze = np.zeros(2)
        self.gaze_target = np.zeros(2)
        self.saccade = np.zeros(2)
        self.next_saccade = 0.5
        self.idle = 10.0
        self.next_look = 0.0
        self.blink_start = -10.0
        self.next_blink = 2.5
        self.light_on = False
        self.light = 0.0
        self.rp = RP
        self.ou = 1.0
        self.dive = 0.0
        self.dive_target = 0.0
        self.speed = 0.05
        self.stars = Stars(3500, self.rng)
        self._shape = None
        self._tex_key = None

    # ---------------------------------------------------------- behaviour

    def look_at(self, wx, wy):
        self.gaze_target = np.clip([wx / 1.5, (wy - 0.02) / 0.9], -1, 1)
        self.idle = 0.0

    def blink(self):
        self.blink_start = self.t

    def toggle_dive(self):
        self.dive_target = 0.0 if self.dive_target > 0.5 else 1.0

    def update(self, dt):
        rng = self.rng
        self.t += dt
        self.idle += dt

        # nobody is moving the mouse: look around on its own
        if self.idle > 4.0 and self.t > self.next_look:
            self.gaze_target = rng.uniform([-0.8, -0.5], [0.8, 0.6])
            self.next_look = self.t + rng.uniform(1.0, 2.8)
        if self.t > self.next_saccade:
            self.saccade = rng.normal(0, 0.025, 2)
            self.next_saccade = self.t + rng.uniform(0.3, 1.4)
        self.gaze += (self.gaze_target + self.saccade - self.gaze) * (1 - math.exp(-dt * 16))

        if self.t > self.next_blink and self.dive < 0.05:
            self.blink()
            self.next_blink = self.t + (0.45 if rng.random() < 0.2 else rng.uniform(2.5, 6.5))
        p = (self.t - self.blink_start) / 0.32
        closure = math.sin(math.pi * p ** 0.6) if 0.0 <= p <= 1.0 else 0.0

        self.light += (float(self.light_on) - self.light) * (1 - math.exp(-dt * 6))
        hippus = 0.006 * math.sin(self.t * 1.7) + 0.004 * math.sin(self.t * 0.63)
        target = RP * (1 - 0.5 * self.light) + hippus
        rate = 8.0 if target < self.rp else 2.5          # constrict fast, dilate slowly
        self.rp += (target - self.rp) * (1 - math.exp(-dt * rate))
        self.ou = (1 - 0.18 * self.light) * (1 - closure)

        step = dt / DIVE_TIME
        if self.dive_target > self.dive:
            self.dive = min(self.dive + step, 1.0)
        else:
            self.dive = max(self.dive - step, 0.0)
        e = float(ss(0.0, 1.0, self.dive))
        self.speed = 0.05 + 1.1 * math.sin(math.pi * self.dive) ** 2 + 0.12 * e
        self.stars.step(dt, self.speed)

    # ---------------------------------------------------------- drawing

    def _cells(self, W, H):
        if self._shape != (H, W):
            self._shape = (H, W)
            self.R0 = self.rng.integers(0, VARIANTS, (H, W))
            self.R1 = self.rng.random((H, W), dtype=np.float32)
            self.R2 = self.rng.random((H, W), dtype=np.float32)

    def _textures(self, x, y, key):
        """Noise that depends only on world position - recomputed only when the camera moves."""
        if self._tex_key != key:
            self._tex_key = key
            blot = bil(TN, x * 40 + 11, y * 40 + 5)
            pore = bil(TW, x * 170 + 3, y * 170 + 9)
            skin = (0.78 + 0.36 * blot) * (0.92 + 0.14 * pore)

            # eyebrow: strands that rise near the nose and flatten outward
            bs = (x + 1.25) / 2.75
            bsc = np.clip(bs, 0, 1)
            bc = 0.66 + 0.18 * np.sin(bsc * np.pi * 0.85) - 0.04 * bsc
            bt = 0.11 * (1 - 0.55 * bsc)
            band = ss(bt, bt * 0.4, np.abs(y - bc)) * ss(-0.02, 0.08, bs) * ss(1.02, 0.85, bs)
            ang = np.radians(75 - 60 * bsc)
            along = x * np.cos(ang) + y * np.sin(ang)
            across = -x * np.sin(ang) + y * np.cos(ang)
            brow = band * ss(0.3, 0.6, bil(TW, along * 6, across * 45))
            brow_phi = np.degrees(ang) + 40 * (bil(TW, along * 6 + 99, across * 45) - 0.5)

            v1 = np.abs(bil(TN, x * 60 + 40, y * 60 + 70) - 0.5)
            v2 = np.abs(bil(TN, x * 130 + 9, y * 130 + 150) - 0.5)
            vein = ss(0.03, 0.0, v1) + 0.6 * ss(0.02, 0.0, v2)
            self._tex = (skin, brow, brow_phi, vein)
        return self._tex

    def _stars(self, W, H, cw, ch, half, zoom, camx, camy, icx, icy, sscale, vp):
        """Splat the star field into per-cell brightness (n,) and colour (n, 3)."""
        st = self.stars
        n = W * H
        acc = np.zeros((4, n), np.float32)
        # a cell covers more sky when the pupil is small, so each star counts for less
        bright = st.mag * (0.7 + 0.3 * np.sin(self.t * 3.0 + st.phase)) * min(1.0, (zoom / sscale) ** 2)
        samples = 1 + int(min(7.0, self.speed * 6))       # more samples = longer warp streaks
        for k in range(samples):
            z = st.z + self.speed * 0.06 * k / samples
            wx = icx + (st.p[:, 0] / z * 0.22 + vp[0]) / sscale
            wy = icy + (st.p[:, 1] / z * 0.22 + vp[1]) / sscale
            j = np.rint((wx - camx) * zoom * half / cw + W / 2 - 0.5).astype(np.int64)
            i = np.rint(-(wy - camy) * zoom * half / ch + H / 2 - 0.5).astype(np.int64)
            ok = (j >= 0) & (j < W) & (i >= 0) & (i < H)
            w = (bright * ss(1.0, 0.45, z) * (1 - k / samples))[ok]
            cell = i[ok] * W + j[ok]
            acc[0] += np.bincount(cell, w, n)
            for c in range(3):
                acc[c + 1] += np.bincount(cell, w * st.col[ok, c], n)
        return acc[0], acc[1:].T

    def _space(self, qx, qy, e):
        """A galaxy and some nebula, in sky coordinates centred on the vanishing point."""
        u = qx * 0.88 + qy * 0.48
        v = (-qx * 0.48 + qy * 0.88) / 0.5
        rg = np.hypot(u, v) + 1e-3
        arms = 0.5 + 0.5 * np.cos(2 * np.arctan2(v, u) - 4.0 * np.log(rg) - self.t * 0.25)
        gal = 1.5 * np.exp(-rg / 0.35) * (0.1 + 0.9 * arms ** 3) * (0.5 + 0.8 * bil(TN, u * 40 + 30, v * 40 + 80))
        core = 1.3 * np.exp(-(rg / 0.05) ** 2) + 0.5 * np.exp(-rg / 0.09)
        col = (mix(rgb(1.0, 0.82, 0.55), rgb(0.55, 0.65, 1.0), ss(0.03, 0.45, rg)) * gal[:, None]
               + rgb(1.0, 0.9, 0.75) * core[:, None])
        neb = ss(0.6, 0.95, bil(TN, qx * 14 + 120, qy * 14 + 40)) ** 2 * 0.5
        col += mix(rgb(0.45, 0.12, 0.55), rgb(0.10, 0.30, 0.60), bil(TN, qx * 9 + 7, qy * 9 + 200)) * neb[:, None]
        return rgb(0.006, 0.006, 0.012) + col * (0.55 + 0.45 * e)

    def frame(self, glyphs, W, H, glow=True):
        """Shade one frame. Returns character ids (H, W) and uint8 fg/bg colours (H, W, 3)."""
        self._cells(W, H)
        cw, ch = glyphs.cw, glyphs.ch
        n = W * H
        e = float(ss(0.0, 1.0, self.dive))
        gx, gy = float(self.gaze[0]), float(self.gaze[1])
        icx, icy = 0.40 * gx, 0.20 * gy + 0.01
        rp = self.rp * (1 + 1.3 * e)
        ou = self.ou

        # ---- camera: falls toward the pupil while diving
        zoom = ZOOM0 * (ZOOM_IN / ZOOM0) ** e
        c = float(ss(0.0, 0.6, self.dive))
        camx = CAM0[0] + (icx - CAM0[0]) * c
        camy = CAM0[1] + (icy - CAM0[1]) * c
        half = H * ch / 2
        sx, sy = np.meshgrid((((np.arange(W) + 0.5) * cw - W * cw / 2) / half).astype(np.float32),
                             (-((np.arange(H) + 0.5) * ch - H * ch / 2) / half).astype(np.float32))
        x, y = camx + sx / zoom, camy + sy / zoom
        skin_tex, brow, brow_phi, vein = self._textures(x, y, (W, H, cw, ch, camx, camy, zoom))

        # ---- lids
        t = x / WO
        tc = np.clip(t, -1, 1)
        at = np.abs(t)
        w = 1 - tc * tc
        base = 0.035 * tc
        lo = base - 0.21 * w ** 1.1 * (1 + 0.12 * tc) * (0.78 + 0.22 * ou)
        up_open = base + (0.40 + 0.06 * gy) * w ** 0.8 * (1 - 0.10 * tc)
        up = lo + (up_open - lo) * ou
        du, dl = up - y, y - lo                     # > 0 inside the opening
        inside = ss(0.0, 0.01, np.minimum(du, dl))
        upper = (y > (up + lo) * 0.5).astype(np.float32)
        lower = 1 - upper
        ho, hl = y - up, lo - y                     # height above / depth below the lids
        near = ss(1.15, 1.0, at)

        # ---- skin
        skin = rgb(0.50, 0.28, 0.19) * skin_tex[..., None]
        t2 = x / (WO * 1.12)
        t2c = np.clip(t2, -1, 1)
        yc = 0.035 * t2c + 0.56 * (1 - t2c * t2c) ** 0.7 + 0.025
        crease_on = ss(1.0, 0.85, np.abs(t2))
        dc = y - yc
        shade = np.ones_like(x)
        shade -= 0.42 * np.exp(-(dc / 0.028) ** 2) * crease_on * (0.5 + 0.5 * ou)
        shade -= 0.18 * np.exp(-((dc + 0.05) / 0.04) ** 2) * crease_on
        shade += 0.14 * np.exp(-((ho - 0.09) / 0.07) ** 2) * upper * (dc < 0)
        shade += 0.12 * np.exp(-((y - 0.95) / 0.15) ** 2)
        shade -= 0.20 * np.exp(-((hl - 0.11) / 0.045) ** 2) * lower
        shade += 0.10 * np.exp(-((hl - 0.05) / 0.03) ** 2) * lower
        shade -= 0.30 * ss(-0.7, -1.7, x) * np.exp(-((y - 0.15) / 0.6) ** 2)
        skin *= shade[..., None]
        liner = np.exp(-(np.maximum(ho, 0) / (0.012 + 0.012 * ss(-0.3, 1.0, tc))) ** 2)
        skin = mix(skin, rgb(0.07, 0.045, 0.035), 0.85 * liner * upper * near)
        skin = mix(skin, rgb(0.78, 0.44, 0.36), 0.6 * np.exp(-(np.maximum(hl, 0) / 0.016) ** 2) * lower * near)

        # ---- lashes: a fan of strands rooted along each lid
        slope = 1.25 * tc + 0.15
        q = (x - slope * ho) * 34
        q += 0.3 * np.sin(q * 0.37) + 0.2 * np.sin(q * 0.91)
        lid = np.floor(q).astype(np.int64)
        length = (0.045 + 0.11 * ss(-1.0, 0.2, tc) * ss(1.02, 0.75, tc)) * (0.65 + 0.7 * TW[lid & 255, 7])
        thick = 0.32 * np.clip(1 - ho / length, 0, 1) + 1e-3
        lash_up = ss(thick, thick * 0.4, np.abs(q - lid - 0.5)) * (ho > 0.004) * ss(length, length * 0.7, ho)
        slope_l = 0.9 * tc + 0.2
        ql = (x + slope_l * hl) * 22
        lidl = np.floor(ql).astype(np.int64)
        length_l = 0.04 * ss(-0.6, 0.2, tc) * ss(1.0, 0.7, tc) * (0.6 + 0.8 * TW[lidl & 255, 31]) + 1e-4
        lash_lo = ss(0.25, 0.1, np.abs(ql - lidl - 0.5)) * (hl > 0.006) * ss(length_l, length_l * 0.6, hl)
        lash_up *= near * upper
        lash_lo *= near * lower
        lash_phi = np.where(lash_up > lash_lo,
                            np.degrees(np.arctan2(1.0, slope)), np.degrees(np.arctan2(-1.0, slope_l)))
        lash = np.maximum(lash_up, lash_lo)
        hair = np.maximum(lash, brow * (1 - inside))
        hair_phi = np.where(lash > brow, lash_phi, brow_phi)
        skin = mix(skin, rgb(0.16, 0.09, 0.06), 0.85 * hair)

        # ---- the sky behind the pupil
        sscale = math.exp(math.log(0.6 / rp) * (1 - e) + math.log(zoom) * e)
        vp = (0.5 * gx * e, 0.35 * gy * e)          # vanishing point steers with the mouse
        star_b, star_rgb = self._stars(W, H, cw, ch, half, zoom, camx, camy, icx, icy, sscale, vp)

        # ---- eyeball: shaded only where the lids are open
        col = skin.reshape(n, 3)
        fam = np.full(n, SKIN, np.int64)
        vis = np.zeros(n, np.float32)              # how much of each cell is open pupil
        ii = np.flatnonzero(inside.ravel() > 0)
        X, Y = x.ravel()[ii], y.ravel()[ii]
        DU, DL, AT = du.ravel()[ii], dl.ravel()[ii], at.ravel()[ii]
        INS = inside.ravel()[ii]

        BY = Y + 0.02
        bz = np.sqrt(np.clip(BALL ** 2 - X * X - BY * BY, 0.02, None))
        lam = np.clip((-0.35 * X + 0.45 * BY + 0.82 * bz) / BALL, 0, 1)
        occ = (1 - 0.6 * np.exp(-np.maximum(DU, 0) / 0.06)) * (1 - 0.30 * np.exp(-np.maximum(DL, 0) / 0.03))
        ball = rgb(0.90, 0.85, 0.80) * (0.50 + 0.50 * lam)[:, None]
        ball = mix(ball, rgb(0.86, 0.55, 0.48), 0.55 * ss(0.45, 1.0, AT))
        ball = mix(ball, rgb(0.75, 0.18, 0.15), np.clip(0.7 * vein.ravel()[ii] * ss(0.3, 0.95, AT), 0, 1))
        fam_e = np.full(ii.size, SCLERA, np.int64)

        # iris, foreshortened as the ball turns
        sqx = math.sqrt(1 - (icx / BALL) ** 2)
        sqy = math.sqrt(1 - ((icy + 0.02) / BALL) ** 2)
        dx, dy = (X - icx) / sqx, (Y - icy) / sqy
        r = np.hypot(dx, dy) / RI
        s = np.flatnonzero(r < 1.05)
        if s.size:
            rs = r[s]
            a = np.arctan2(dy[s], dx[s]) / (2 * np.pi)
            f1 = bil(TN, a * 768, rs * 14)
            f2 = bil(TN, a * 2304 + 91, rs * 32 + 40)
            fib = ss(0.28, 0.80, 0.55 * f1 + 0.45 * f2)
            rpn = rp / RI * (1 + 0.04 * (bil(TN, a * 512, np.full_like(a, 7.0)) - 0.5))
            iris = mix(rgb(1.00, 0.72, 0.30), rgb(0.70, 0.36, 0.13), ss(rpn + 0.04, 0.60, rs))
            iris = mix(iris, rgb(0.32, 0.14, 0.05), ss(0.60, 0.97, rs))
            iris *= (0.30 + 1.0 * fib)[:, None]
            collar = np.exp(-((rs - rpn - 0.17 - 0.06 * (f1 - 0.5)) / 0.05) ** 2)
            iris += rgb(0.30, 0.20, 0.07) * collar[:, None]
            crypt = (ss(0.66, 0.74, bil(TN, a * 1024 + 33, rs * 10 + 11))
                     * ss(rpn + 0.08, rpn + 0.22, rs) * ss(0.92, 0.75, rs))
            iris *= ((1 - 0.65 * crypt) * (1 - 0.85 * ss(0.80, 1.0, rs)) * (0.75 + 0.25 * lam[s]))[:, None]
            iris_m = ss(1.02, 0.97, rs)
            pupil_m = ss(rpn + 0.02, rpn - 0.02, rs)
            ball[s] = mix(ball[s], iris, iris_m)
            fam_e[s[iris_m > 0.5]] = IRIS

            # the pupil is a window into space
            p = np.flatnonzero(pupil_m > 0)
            if p.size:
                sp = s[p]
                qx = (X[sp] - icx) * sscale - vp[0]
                qy = (Y[sp] - icy) * sscale - vp[1]
                space = self._space(qx, qy, e) + star_rgb[ii[sp]]
                ball[sp] = mix(ball[sp], space, pupil_m[p])
                fam_e[sp[pupil_m[p] > 0.5]] = SPACE
                vis[ii[sp]] = pupil_m[p] * INS[sp]

        ball *= occ[:, None]
        ball = mix(ball, rgb(0.80, 0.42, 0.36), 0.65 * np.exp(-(np.maximum(DL, 0) / 0.02) ** 2))
        car = np.exp(-(((X + WO * 0.90) / 0.06) ** 2 + ((Y + 0.0315) / 0.045) ** 2))
        ball = mix(ball, rgb(0.84, 0.45, 0.40), 0.9 * car)
        # reflection of a window on the cornea, plus a small fill light
        hx, hy = icx - 0.13, icy + 0.13
        ex, ey = ((X - hx) / 0.055) ** 2, ((Y - hy) / 0.06) ** 2
        win = np.exp(-(ex * ex + ey * ey) * 1.5)
        dot = np.exp(-(((X - icx - 0.15) / 0.018) ** 2 + ((Y - icy + 0.13) / 0.016) ** 2))
        spec = (0.95 * win + 0.6 * dot) * ss(1.2, 1.0, r) * (1 - e)
        ball = mix(ball, rgb(1.0, 0.97, 0.92), np.clip(spec, 0, 1))
        fam_e[spec > 0.4] = SCLERA

        col[ii] = mix(col[ii], ball, INS)
        fam[ii] = np.where(INS > 0.5, fam_e, SKIN)
        col = col.reshape(H, W, 3)
        fam = fam.reshape(H, W)

        vig = 1 - 0.55 * ss(0.7, 1.6, np.hypot(sx * 0.6, sy))
        col *= (vig * (1 + 0.15 * self.light))[..., None]
        np.clip(col, 0, None, out=col)

        # ---- choose characters
        L = col @ np.array([0.299, 0.587, 0.114], np.float32)
        level = np.clip(np.rint(np.clip(L, 0, 1) ** 0.9 * 1.05 * (LEVELS - 1) - 0.4), 0, LEVELS - 1).astype(np.int64)
        var = (self.R0 + np.floor(self.t * 0.4 + self.R1).astype(np.int64)) % VARIANTS
        idx = glyphs.lut[fam, level, var]

        # strokes: hair follows its growth direction, skin edges follow their contour
        Lp = np.pad(L, 1, mode="edge")
        Lb = (Lp[:-2, 1:-1] + Lp[2:, 1:-1] + Lp[1:-1, :-2] + Lp[1:-1, 2:] + 2 * L) / 6   # soften pores
        gxL = np.gradient(Lb, axis=1)
        gyL = -np.gradient(Lb, axis=0) * (cw / ch)
        edge = (fam == SKIN) & (gxL * gxL + gyL * gyL > 0.07 ** 2) & (level > 0)
        idx[edge] = glyphs.strokes[stroke_index(np.degrees(np.arctan2(gyL[edge], gxL[edge])) + 90)]
        strand = (hair > 0.45) & (fam == SKIN)
        idx[strand] = glyphs.strokes[stroke_index(hair_phi[strand])]

        # stars: dots when drifting, streaks pointing away from the vanishing point when warping
        star_b = (star_b * vis).reshape(H, W)
        star = star_b > 0.12
        if star.any():
            if self.speed > 0.35:
                qx = (x[star] - icx) * sscale - vp[0]
                qy = (y[star] - icy) * sscale - vp[1]
                idx[star] = glyphs.strokes[stroke_index(np.degrees(np.arctan2(qy, qx)))]
            else:
                b = star_b[star]
                idx[star] = glyphs.star[(b > 0.45).astype(np.int64) + (b > 0.9)]

        # ---- colours: glyph density carries brightness, so keep the hue vivid
        mx = col.max(-1, keepdims=True)
        disp = col / np.maximum(mx, 1e-4) * (0.12 + 0.88 * np.clip(mx, 0, 1) ** 0.75)
        fg = np.clip(disp * 255, 0, 255).astype(np.uint8)
        bg = np.clip(col * (0.10 * 255), 0, 255).astype(np.uint8) if glow else None

        # characters rain into place when the program starts
        if self.t < 2.0:
            hidden = self.R2 > self.t / 1.6
            idx = np.where(hidden, self.rng.integers(1, 95, (H, W)), idx)
            fg = np.where(hidden[..., None], fg // 3, fg)
            if bg is not None:
                bg = np.where(hidden[..., None], 0, bg).astype(np.uint8)
        return idx, fg, bg


def render(eye, glyphs, W, H, glow=True):
    return glyphs.compose(*eye.frame(glyphs, W, H, glow))


# ------------------------------------------------------------------ app

HELP = ("mouse: look   hold click: light   space: dive in / out   B: blink   "
        "+/-: size   G: glow   S: screenshot   H: help   Esc: quit")


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    pygame.init()
    pygame.display.set_caption("Eye -> Space")
    windowed = (1280, 760)
    screen = pygame.display.set_mode(windowed, pygame.RESIZABLE)
    font_px = 13
    glyphs = Glyphs(font_px)
    hud = pygame.font.SysFont("consolas,couriernew", 14)
    eye = Eye()
    clock = pygame.time.Clock()
    glow, help_until, fullscreen = True, 8.0, False

    while True:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        sw, sh = screen.get_size()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    return
                elif k == pygame.K_SPACE:
                    eye.toggle_dive()
                elif k == pygame.K_b:
                    eye.blink()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    font_px = min(font_px + 1, 28)
                    glyphs = Glyphs(font_px)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    font_px = max(font_px - 1, 7)
                    glyphs = Glyphs(font_px)
                elif k == pygame.K_g:
                    glow = not glow
                elif k == pygame.K_h:
                    help_until = 0.0 if eye.t < help_until else eye.t + 1e9
                elif k == pygame.K_s:
                    name = time.strftime("eye_%Y%m%d_%H%M%S.png")
                    pygame.image.save(screen, name)
                    print("saved", name)
                elif k == pygame.K_F11:
                    fullscreen = not fullscreen
                    screen = (pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if fullscreen
                              else pygame.display.set_mode(windowed, pygame.RESIZABLE))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                eye.light_on = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                eye.light_on = False
            elif ev.type == pygame.MOUSEMOTION:
                eye.look_at(CAM0[0] + (ev.pos[0] - sw / 2) / (sh / 2) / ZOOM0,
                            CAM0[1] - (ev.pos[1] - sh / 2) / (sh / 2) / ZOOM0)

        eye.update(dt)
        sw, sh = screen.get_size()
        W, H = max(sw // glyphs.cw, 1), max(sh // glyphs.ch, 1)
        screen.fill((0, 0, 0))
        screen.blit(render(eye, glyphs, W, H, glow), ((sw - W * glyphs.cw) // 2, (sh - H * glyphs.ch) // 2))
        if eye.t < help_until:
            text = hud.render(HELP, True, (150, 150, 150), (0, 0, 0))
            text.set_alpha(int(255 * min(1.0, help_until - eye.t)))
            screen.blit(text, (12, sh - text.get_height() - 10))
        pygame.display.flip()


if __name__ == "__main__":
    main()
