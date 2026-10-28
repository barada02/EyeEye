"""In Your Eyes - a journey drawn entirely with text characters.

Look into an eye. Fall through its pupil into a galaxy, toward one small star,
its planets, the Earth, a single light on the night side, the home it belongs
to ... and what is waiting inside. Then it begins again.

Controls
  mouse         the eye follows the cursor
  scroll        scrub / travel smoothly between worlds
  hold click    shine a light - the pupil constricts
  space         travel deeper
  backspace     travel back out
  1 - 6         jump directly to a world
  A             autoplay the whole journey
  P             pause / resume
  C             cycle eye style (natural blue, emerald, original)
  M             toggle ambient sound
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

from earth_map import land_mask
try:
    from sound import SoundEngine
except ImportError:
    class SoundEngine:
        def __init__(self): self.enabled = False
        def toggle_mute(self): return True
        def update(self, *a): pass

LEVELS = 20      # brightness steps in each character ramp
VARIANTS = 6     # interchangeable characters per step (cells drift between them)

# Every kind of surface is written in its own alphabet.
FAMILIES = (
    "`.',:;-~!ilrjftc1/\\|()?7ivxzsJLYTCZunoe",              # skin
    "`.'-:;~=+*^ocszaxe%&#@",                                # sclera, clouds, moonlight
    "`.':;ilrjcvunxzoaeJLCUYXZOQ0DGBRKHMNWAPSE8%$&#@",      # iris
    "`.',-:;~=oO0@",                                         # space (stars get their own glyphs)
    "`.:-=+*oO0Q@#%&",                                       # suns, lamps, city lights
    "`.'-~=:;^\"vw",                                         # sea
    "`.,:;!ilxXkKHM#%@&",                                    # land
    "`.,:;-=|!iIl1HN#%@",                                    # house
    "`.,:;-~vVeEoO<3*",                                      # heart
)
SKIN, SCLERA, IRIS, SPACE, SUN, SEA, LAND, HOUSE, HEART = range(len(FAMILIES))
EDGE_FAMILIES = (SKIN, HOUSE)   # contours in these get drawn with line strokes
STROKES = "-/|\\"               # line glyphs for 0, 45, 90, 135 degrees

TRAVEL_TIME = 2.6   # seconds to fall from one world into the next
HOLD_TIME = 3.5     # seconds autoplay lingers in each world

# the eye
WO = 1.0            # half-width of the eye opening (world units; screen half-height = 1)
RI = 0.36           # iris radius
RP = 0.13           # resting pupil radius
BALL = 1.25         # eyeball radius


# ------------------------------------------------------------------ helpers

def ss(a, b, x):
    """Smoothstep; a > b gives the falling version."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a + (b - a) * t[..., None]


def rgb(r, g, b):
    return np.array([r, g, b], np.float32)


def fill(k, colour):
    out = np.empty((k, 3), np.float32)
    out[:] = colour
    return out


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
    """Bilinear, wrapping lookup of a power-of-two sized texture T at texel coords (x, y)."""
    h, w = T.shape
    flat = T.ravel()
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    x0 = xf.astype(np.int32) & (w - 1)
    x1 = (x0 + 1) & (w - 1)
    y0 = (yf.astype(np.int32) & (h - 1)) * w
    y1 = (y0 + w) & (h * w - 1)
    a, b = flat.take(y0 + x0), flat.take(y0 + x1)
    c, d = flat.take(y1 + x0), flat.take(y1 + x1)
    top = a + (b - a) * fx
    return top + (c + (d - c) * fx - top) * fy


def stroke_index(phi):
    """Line orientation in degrees (world space, y up) -> index into STROKES."""
    return np.rint((phi % 180.0) / 45.0).astype(np.int64) % 4


def unit(*v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


_rng = np.random.default_rng(7)
TN = tile_noise(256, 6, _rng)                      # smooth fractal noise
TW = _rng.random((256, 256), dtype=np.float32)     # white noise
MAP = land_mask(512, 256)                          # the Earth's continents


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
        self.strokes = self.text_ids(STROKES)
        self.star = self.text_ids(".+*")

    def text_ids(self, s):
        return np.array([self.index[c] for c in s])

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


# ------------------------------------------------------------------ shared pieces

class Stars:
    """A 3D star field flying toward the camera; shared by every world in space."""
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


class Frame:
    """What every world needs while shading one frame: the grid, glyphs and clocks."""

    def __init__(self, journey, glyphs, W, H):
        self.j, self.g, self.t = journey, glyphs, journey.t
        self.W, self.H, self.n = W, H, W * H
        self.cw, self.ch = glyphs.cw, glyphs.ch
        self.half = H * glyphs.ch / 2

    def splat(self, px, py, w, colours, T):
        """Points in a world's screen coords -> (4, n) brightness + RGB accumulated per screen cell."""
        a, bx, by = T
        j = np.rint((a * px + bx) * self.half / self.cw + self.W / 2 - 0.5).astype(np.int64)
        i = np.rint(-(a * py + by) * self.half / self.ch + self.H / 2 - 0.5).astype(np.int64)
        ok = (j >= 0) & (j < self.W) & (i >= 0) & (i < self.H)
        cell, w = i[ok] * self.W + j[ok], w[ok]
        acc = np.empty((4, self.n), np.float32)
        acc[0] = np.bincount(cell, w, self.n)
        for c in range(3):
            acc[c + 1] = np.bincount(cell, w * colours[ok, c], self.n)
        return acc

    def sparkles(self, b):
        return self.g.star[(b > 0.45).astype(np.int64) + (b > 0.9)]

    def starfield(self, sx, sy, cells, T):
        """Star colour to add (k, 3) and glyph overrides (k,) for the given cells."""
        st, speed = self.j.stars, self.j.speed
        # a cell covers more sky when this world is small on screen, so each star counts for less
        bright = st.mag * (0.7 + 0.3 * np.sin(self.t * 3.0 + st.phase)) * min(1.0, T[0] ** 2)
        samples = 1 + int(min(7.0, speed * 6))           # more samples = longer warp streaks
        acc = np.zeros((4, self.n), np.float32)
        for k in range(samples):
            z = st.z + speed * 0.06 * k / samples
            w = bright * ss(1.0, 0.45, z) * (1 - k / samples)
            acc += self.splat(st.p[:, 0] / z * 0.22, st.p[:, 1] / z * 0.22, w, st.col, T)
        b = acc[0, cells]
        ov = np.full(cells.size, -1, np.int64)
        on = b > 0.12
        if on.any():
            if speed > 0.35:      # streaks point away from where we are heading
                ov[on] = self.g.strokes[stroke_index(np.degrees(np.arctan2(sy[on], sx[on])))]
            else:
                ov[on] = self.sparkles(b[on])
        return acc[1:, cells].T, ov

    def text(self, cells, word):
        """Write `word` repeatedly across the given cells, brick-offset row by row."""
        ids = self.g.text_ids(word)
        i, j = np.divmod(cells, self.W)
        return ids[(j + 3 * i) % ids.size]


def blend_inner(col, fam, ov, pm, inner):
    """Show the next world through portal mask `pm`."""
    got = inner(pm)
    if got is not None:
        sel, ncol, nfam, nov = got
        m = pm[sel]
        col[sel] = mix(col[sel], ncol, m)
        take = m > 0.5
        fam[sel[take]] = nfam[take]
        ov[sel[take]] = nov[take]


# ------------------------------------------------------------------ worlds
#
# Each world has a default camera (zoom0, cam0) and a portal: a point where the
# next world shows through. While travelling the camera falls toward the portal
# (up to zoom_end) and the portal grows until the next world fills the screen at
# exactly its own default view.  shade() gets world coords (x, y), the world's
# own screen coords (sx, sy), the screen cells being drawn, T (this world's
# screen -> real screen: s = a * s' + b) and `inner`, which draws the next world.

class EyeWorld:
    zoom0, cam0, zoom_end = 1.35, (0.0, 0.10), 14.0

    def __init__(self, rng):
        self.rng = rng
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
        self._tex_key = None

    def look_at(self, wx, wy):
        self.gaze_target = np.clip([wx / 1.5, (wy - 0.02) / 0.9], -1, 1)
        self.idle = 0.0

    def blink(self):
        self.blink_start = self.t

    def update(self, dt, busy):
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

        if self.t > self.next_blink and not busy:
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

    def portal(self, fr, e):
        return 0.40 * float(self.gaze[0]), 0.20 * float(self.gaze[1]) + 0.01, 0.6 / self.rp

    def _textures(self, x, y):
        """Noise that depends only on position - recomputed only when the view changes."""
        key = (x.size, float(x[0]), float(x[-1]), float(y[0]), float(y[-1]),
               float(x[x.size // 2]), float(y[y.size // 2]))
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

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        k = x.size
        gy = float(self.gaze[1])
        icx, icy, _ = self.portal(fr, e)
        rp = self.rp * (1 + 1.3 * e)
        ou = self.ou
        skin_tex, brow, brow_phi, vein = self._textures(x, y)

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

        theme = getattr(fr.j, "eye_theme", 0)

        # ---- skin and eye palettes
        if theme == 0:    # Natural Human (Ocean Blue & Natural Fair/Peach Skin)
            base_skin = rgb(0.86, 0.68, 0.56)
            crease_shade = 0.35
            liner_col = rgb(0.12, 0.08, 0.07)
            waterline_col = rgb(0.86, 0.52, 0.46)
            hair_col = rgb(0.18, 0.11, 0.08)
            sclera_col = rgb(0.96, 0.95, 0.94)
            vein_col = rgb(0.80, 0.22, 0.18)
            caruncle_col = rgb(0.88, 0.52, 0.46)
        elif theme == 1:  # Natural Human (Emerald Green & Warm Skin)
            base_skin = rgb(0.84, 0.67, 0.54)
            crease_shade = 0.37
            liner_col = rgb(0.10, 0.07, 0.06)
            waterline_col = rgb(0.85, 0.50, 0.44)
            hair_col = rgb(0.14, 0.09, 0.06)
            sclera_col = rgb(0.95, 0.94, 0.93)
            vein_col = rgb(0.78, 0.20, 0.16)
            caruncle_col = rgb(0.86, 0.48, 0.43)
        else:             # Original (Bronze Skin & Amber Iris)
            base_skin = rgb(0.50, 0.28, 0.19)
            crease_shade = 0.42
            liner_col = rgb(0.07, 0.045, 0.035)
            waterline_col = rgb(0.78, 0.44, 0.36)
            hair_col = rgb(0.16, 0.09, 0.06)
            sclera_col = rgb(0.90, 0.85, 0.80)
            vein_col = rgb(0.75, 0.18, 0.15)
            caruncle_col = rgb(0.84, 0.45, 0.40)

        # ---- skin
        skin = base_skin * skin_tex[..., None]
        t2 = x / (WO * 1.12)
        t2c = np.clip(t2, -1, 1)
        yc = 0.035 * t2c + 0.56 * (1 - t2c * t2c) ** 0.7 + 0.025
        crease_on = ss(1.0, 0.85, np.abs(t2))
        dc = y - yc
        shade = np.ones_like(x)
        shade -= crease_shade * np.exp(-(dc / 0.028) ** 2) * crease_on * (0.5 + 0.5 * ou)
        shade -= 0.18 * np.exp(-((dc + 0.05) / 0.04) ** 2) * crease_on
        shade += 0.14 * np.exp(-((ho - 0.09) / 0.07) ** 2) * upper * (dc < 0)
        shade += 0.12 * np.exp(-((y - 0.95) / 0.15) ** 2)
        shade -= 0.20 * np.exp(-((hl - 0.11) / 0.045) ** 2) * lower
        shade += 0.10 * np.exp(-((hl - 0.05) / 0.03) ** 2) * lower
        shade -= 0.25 * ss(-0.7, -1.7, x) * np.exp(-((y - 0.15) / 0.6) ** 2)
        skin *= shade[..., None]
        liner = np.exp(-(np.maximum(ho, 0) / (0.012 + 0.012 * ss(-0.3, 1.0, tc))) ** 2)
        skin = mix(skin, liner_col, 0.85 * liner * upper * near)
        skin = mix(skin, waterline_col, 0.6 * np.exp(-(np.maximum(hl, 0) / 0.016) ** 2) * lower * near)

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
        skin = mix(skin, hair_col, 0.85 * hair)

        col = skin
        fam = np.full(k, SKIN, np.int64)
        ov = np.full(k, -1, np.int64)
        pm = np.zeros(k, np.float32)

        # ---- eyeball: shaded only where the lids are open
        ii = np.flatnonzero(inside > 0)
        X, Y = x[ii], y[ii]
        DU, DL, AT, INS = du[ii], dl[ii], at[ii], inside[ii]

        BY = Y + 0.02
        bz = np.sqrt(np.clip(BALL ** 2 - X * X - BY * BY, 0.02, None))
        lam = np.clip((-0.35 * X + 0.45 * BY + 0.82 * bz) / BALL, 0, 1)
        occ = (1 - 0.6 * np.exp(-np.maximum(DU, 0) / 0.06)) * (1 - 0.30 * np.exp(-np.maximum(DL, 0) / 0.03))
        ball = sclera_col * (0.50 + 0.50 * lam)[:, None]
        ball = mix(ball, rgb(0.86, 0.55, 0.48), 0.50 * ss(0.45, 1.0, AT))
        ball = mix(ball, vein_col, np.clip(0.6 * vein[ii] * ss(0.3, 0.95, AT), 0, 1))
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
            if theme == 0:    # Natural Human (Ocean Blue & Hazel Ring)
                iris = mix(rgb(0.92, 0.72, 0.35), rgb(0.24, 0.64, 0.90), ss(rpn + 0.03, 0.50, rs))
                iris = mix(iris, rgb(0.10, 0.26, 0.48), ss(0.55, 0.98, rs))
                iris *= (0.35 + 0.95 * fib)[:, None]
                collar = np.exp(-((rs - rpn - 0.17 - 0.06 * (f1 - 0.5)) / 0.05) ** 2)
                iris += rgb(0.45, 0.32, 0.12) * collar[:, None]
                crypt = (ss(0.66, 0.74, bil(TN, a * 1024 + 33, rs * 10 + 11))
                         * ss(rpn + 0.08, rpn + 0.22, rs) * ss(0.92, 0.75, rs))
                iris *= ((1 - 0.60 * crypt) * (1 - 0.80 * ss(0.80, 1.0, rs)) * (0.75 + 0.25 * lam[s]))[:, None]
            elif theme == 1:  # Natural Human (Emerald Green & Golden Hazel)
                iris = mix(rgb(0.90, 0.70, 0.30), rgb(0.24, 0.66, 0.38), ss(rpn + 0.03, 0.50, rs))
                iris = mix(iris, rgb(0.10, 0.30, 0.18), ss(0.55, 0.98, rs))
                iris *= (0.35 + 0.95 * fib)[:, None]
                collar = np.exp(-((rs - rpn - 0.17 - 0.06 * (f1 - 0.5)) / 0.05) ** 2)
                iris += rgb(0.42, 0.30, 0.10) * collar[:, None]
                crypt = (ss(0.66, 0.74, bil(TN, a * 1024 + 33, rs * 10 + 11))
                         * ss(rpn + 0.08, rpn + 0.22, rs) * ss(0.92, 0.75, rs))
                iris *= ((1 - 0.60 * crypt) * (1 - 0.80 * ss(0.80, 1.0, rs)) * (0.75 + 0.25 * lam[s]))[:, None]
            else:             # Original (Bronze & Amber)
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
            # the pupil: black, and a window into the galaxy (dim until we fall in)
            ball[s] = mix(ball[s], rgb(0.005, 0.005, 0.01), pupil_m)
            fam_e[s[pupil_m > 0.5]] = SPACE
            pm[ii[s]] = pupil_m * INS[s] * (0.55 + 0.45 * e)

        ball *= occ[:, None]
        ball = mix(ball, rgb(0.80, 0.42, 0.36), 0.65 * np.exp(-(np.maximum(DL, 0) / 0.02) ** 2))
        car = np.exp(-(((X + WO * 0.90) / 0.06) ** 2 + ((Y + 0.0315) / 0.045) ** 2))
        ball = mix(ball, caruncle_col, 0.9 * car)
        col[ii] = mix(col[ii], ball, INS)
        fam[ii] = np.where(INS > 0.5, fam_e, SKIN)

        blend_inner(col, fam, ov, pm, inner)

        # reflection of a window on the cornea, on top of everything in the pupil
        hx, hy = icx - 0.13, icy + 0.13
        ex, ey = ((X - hx) / 0.055) ** 2, ((Y - hy) / 0.06) ** 2
        win = np.exp(-(ex * ex + ey * ey) * 1.5)
        dot = np.exp(-(((X - icx - 0.15) / 0.018) ** 2 + ((Y - icy + 0.13) / 0.016) ** 2))
        spec = (0.95 * win + 0.6 * dot) * ss(1.2, 1.0, r) * (1 - e)
        col[ii] = mix(col[ii], rgb(1.0, 0.97, 0.92), np.clip(spec, 0, 1) * INS)
        hs = ii[(spec > 0.4) & (INS > 0.5)]
        fam[hs] = SCLERA
        ov[hs] = -1

        strand = (hair > 0.45) & (fam == SKIN)
        ov[strand] = fr.g.strokes[stroke_index(hair_phi[strand])]
        return col, fam, ov


def galaxy_sky(x, y, t):
    """A tilted spiral galaxy with a little nebula around it."""
    u = x * 0.88 + y * 0.48
    v = (-x * 0.48 + y * 0.88) / 0.5
    rg = np.hypot(u, v) + 1e-3
    arms = 0.5 + 0.5 * np.cos(2 * np.arctan2(v, u) - 4.0 * np.log(rg) - t * 0.25)
    gal = 1.5 * np.exp(-rg / 0.35) * (0.1 + 0.9 * arms ** 3) * (0.5 + 0.8 * bil(TN, u * 40 + 30, v * 40 + 80))
    core = 1.3 * np.exp(-(rg / 0.05) ** 2) + 0.5 * np.exp(-rg / 0.09)
    col = (mix(rgb(1.0, 0.82, 0.55), rgb(0.55, 0.65, 1.0), ss(0.03, 0.45, rg)) * gal[:, None]
           + rgb(1.0, 0.9, 0.75) * core[:, None])
    neb = ss(0.6, 0.95, bil(TN, x * 14 + 120, y * 14 + 40)) ** 2 * 0.5
    col += mix(rgb(0.45, 0.12, 0.55), rgb(0.10, 0.30, 0.60), bil(TN, x * 9 + 7, y * 9 + 200)) * neb[:, None]
    return rgb(0.006, 0.006, 0.012) + col


class GalaxyWorld:
    zoom0, cam0, zoom_end = 1.0, (0.0, 0.0), 25.0
    STAR = (0.42, -0.16)          # one ordinary star out on a spiral arm

    def portal(self, fr, e):
        return self.STAR[0], self.STAR[1], 10.0

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        col = galaxy_sky(x, y, fr.t)
        fam = np.full(x.size, SPACE, np.int64)
        add, ov = fr.starfield(sx, sy, cells, T)
        col += add

        px, py = self.STAR
        d = np.hypot(x - px, y - py)
        r_in = 0.012 * (1 + 7 * e)
        blend_inner(col, fam, ov, ss(r_in * 3, r_in, d), inner)

        flare = (1.4 * np.exp(-(d / 0.018) ** 2) + 0.35 * np.exp(-d / 0.05)
                 + 0.5 * np.exp(-np.abs(x - px) / 0.004 - np.abs(y - py) / 0.07)
                 + 0.5 * np.exp(-np.abs(y - py) / 0.006 - np.abs(x - px) / 0.12))
        glow = flare * (0.85 + 0.15 * math.sin(fr.t * 2.3)) * (1 - e) ** 2
        col += rgb(1.0, 0.85, 0.6) * glow[:, None]
        hot = glow > 0.6
        fam[hot] = SUN
        ov[hot] = -1
        return col, fam, ov


TILT = 0.36   # how flattened the orbits look
PLANETS = (   # orbit radius, size, colour, period (s), starting angle
    (0.22, 0.018, (0.62, 0.58, 0.55), 6.0, 0.3),
    (0.32, 0.026, (0.93, 0.80, 0.55), 10.0, 2.0),
    (0.45, 0.040, (0.25, 0.50, 0.95), 16.0, 4.4),      # Earth
    (0.58, 0.022, (0.85, 0.38, 0.22), 26.0, 1.1),
    (0.82, 0.060, (0.85, 0.66, 0.45), 50.0, 3.0),
    (1.08, 0.050, (0.90, 0.80, 0.55), 80.0, 5.2),      # ringed
    (1.32, 0.035, (0.55, 0.85, 0.90), 120.0, 0.8),
    (1.56, 0.035, (0.30, 0.45, 0.95), 170.0, 2.6),
)
EARTH = 2


class SolarWorld:
    zoom0, cam0, zoom_end = 1.0, (0.0, 0.0), 25.0

    @staticmethod
    def planet_pos(i, clock):
        a, _, _, period, phase = PLANETS[i]
        th = phase + clock * 2 * math.pi / period
        return a * math.cos(th), a * math.sin(th) * TILT, math.sin(th) > 0

    def portal(self, fr, e):
        ex, ey, _ = self.planet_pos(EARTH, fr.j.orbit_t)
        return ex, ey, EarthWorld.R / PLANETS[EARTH][1]

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        k = x.size
        col = fill(k, rgb(0.004, 0.004, 0.010))
        fam = np.full(k, SPACE, np.int64)
        add, ov = fr.starfield(sx, sy, cells, T)
        col += add

        rr = np.hypot(x, y / TILT) + 1e-6
        grad = np.hypot(x, y / TILT ** 2) / rr               # turns |rr - a| into a distance
        orbit = sum(np.exp(-((rr - p[0]) / grad / 0.016) ** 2) for p in PLANETS)
        col += rgb(0.30, 0.36, 0.55) * (0.5 * np.minimum(orbit, 1))[:, None]

        # the sun
        d = np.hypot(x, y)
        sun = ss(0.092, 0.084, d)
        surf = (0.8 + 0.4 * bil(TN, x * 90 + fr.t * 4, y * 90)) * (1 - 0.35 * (d / 0.09) ** 2)
        col = mix(col, rgb(1.0, 0.80, 0.35) * surf[:, None], sun)
        ang = np.arctan2(y, x)
        rays = 0.6 + 0.4 * np.sin(ang * 9 + fr.t * 0.7) * np.sin(ang * 5 - fr.t * 0.4)
        corona = np.exp(-np.maximum(d - 0.09, 0) / 0.05) * (1 - sun) * rays * 0.9
        col += rgb(1.0, 0.55, 0.18) * corona[:, None]
        hot = (sun > 0.5) | (corona > 0.35)
        fam[hot] = SUN
        ov[hot] = -1

        # planets, lit from the sun
        for i, (a, R, c, _, _) in enumerate(PLANETS):
            px, py, behind = self.planet_pos(i, fr.j.orbit_t)
            dx, dy = x - px, y - py
            disc = ss(R + 0.006, R - 0.006, np.hypot(dx, dy))
            if behind and math.hypot(px, py) < 0.1:
                disc = disc * ss(0.085, 0.1, d)
            on = np.flatnonzero(disc > 0)
            if not on.size:
                continue
            nx, ny = dx[on] / R, dy[on] / R
            nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
            l = math.hypot(px, py) + 1e-6
            lam = np.clip(-(nx * px + ny * py) / l * 0.85 + 0.35 * nz, 0, 1)
            pc = rgb(*c) * (0.10 + 0.9 * lam)[:, None]
            if i == 4:      # banded giant
                pc *= (0.8 + 0.25 * np.sin(ny * 14))[:, None]
            col[on] = mix(col[on], pc, disc[on])
            body = on[disc[on] > 0.5]
            fam[body] = SEA if i == EARTH else SUN
            ov[body] = -1
            if i == 5:      # rings
                rr2 = np.hypot(dx, dy / 0.3) / R
                ring = ss(1.35, 1.5, rr2) * ss(2.3, 2.15, rr2) * ~((dy > 0) & (disc > 0.5))
                col = mix(col, rgb(0.62, 0.55, 0.40), 0.8 * ring)
                fam[ring > 0.5] = SCLERA
                ov[ring > 0.5] = -1

        # the Earth opens up into a close-up of itself
        ex, ey, _ = self.planet_pos(EARTH, fr.j.orbit_t)
        R = PLANETS[EARTH][1]
        de = np.hypot(x - ex, y - ey)
        r_in = R * (1.05 + 3 * e)
        blend_inner(col, fam, ov, ss(r_in + R * 0.6, r_in, de), inner)
        # a faint pulsing ring: this is the one
        ring = np.exp(-((de - R * 2.2) / 0.012) ** 2) * (0.5 + 0.3 * math.sin(fr.t * 3)) * (1 - e) ** 2
        col += rgb(0.35, 0.65, 1.0) * ring[:, None]
        return col, fam, ov


class EarthWorld:
    zoom0, cam0, zoom_end = 1.0, (0.0, 0.0), 30.0
    R = 0.62
    HOME = (0.44, 0.08)                     # one light, just past the edge of night
    LIGHT = unit(-0.75, 0.3, 0.6)

    def portal(self, fr, e):
        return self.HOME[0], self.HOME[1], 25.0

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        k = x.size
        col = fill(k, rgb(0.004, 0.004, 0.010))
        fam = np.full(k, SPACE, np.int64)
        add, ov = fr.starfield(sx, sy, cells, T)
        col += add

        R = self.R
        lx, ly, lz = self.LIGHT
        d = np.hypot(x, y)
        facing = ss(-0.4, 0.6, (x * lx + y * ly) / np.maximum(d, 1e-3))
        halo = np.exp(-np.maximum(d - R, 0) / 0.035) * (d > R * 0.98) * (0.15 + 0.85 * facing)
        col += rgb(0.30, 0.55, 1.0) * (0.7 * halo)[:, None]

        g = np.flatnonzero(d < R)
        if g.size:
            X, Y = x[g] / R, y[g] / R
            Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, 1))
            lam = X * lx + Y * ly + Z * lz
            # tilt the axis, spin, look up the map
            ct, st = math.cos(0.41), math.sin(0.41)
            ax, ay = X * ct - Y * st, X * st + Y * ct
            lat = np.arcsin(np.clip(ay, -1, 1))
            lon = np.arctan2(ax, Z) + fr.j.earth_t * 0.12
            u = lon / (2 * np.pi) * 512
            v = np.clip((0.5 - lat / np.pi) * 256, 0, 254.99)
            m = bil(MAP, u, v) + 0.22 * (bil(TN, u, v + 17) - 0.63)     # 0.63 = TN's median
            land = ss(0.46, 0.54, m)
            alat = np.abs(np.degrees(lat))

            ocean = mix(rgb(0.03, 0.12, 0.40), rgb(0.08, 0.40, 0.62), ss(0.12, 0.46, m))
            dry = ss(0.62, 0.82, bil(TN, u * 2 + 40, v * 2 + 9)) * ss(10, 20, alat) * ss(40, 30, alat)
            green = mix(rgb(0.14, 0.46, 0.14), rgb(0.38, 0.52, 0.18), bil(TN, u * 3 + 5, v * 3))
            surf = mix(ocean, mix(green, rgb(0.72, 0.58, 0.32), dry), land)
            ice = ss(64, 72, alat + 8 * (bil(TN, u * 2 + 9, v * 2 + 50) - 0.5))
            surf = mix(surf, rgb(0.90, 0.93, 0.97), ice)
            cloud = ss(0.78, 0.92, bil(TN, u * 1.5 + fr.t * 1.5, v * 1.5 + 31))

            day = np.clip(lam, 0, 1)
            surf = surf * (0.035 + 0.965 * day)[:, None]
            surf = mix(surf, rgb(0.95, 0.95, 0.97) * (0.06 + 0.94 * day)[:, None], 0.85 * cloud)
            hx, hy, hz = unit(lx, ly, lz + 1)
            glint = np.clip(X * hx + Y * hy + Z * hz, 0, 1) ** 60 * (1 - land) * (1 - cloud)
            surf += 0.8 * glint[:, None]
            night = ss(0.12, -0.05, lam)
            city = land * night * ss(0.86, 0.97, bil(TW, u * 0.5 + 7, v * 0.5 + 3)) * ss(62, 50, alat)
            surf += rgb(1.0, 0.72, 0.32) * (1.2 * city * (1 - 0.7 * cloud))[:, None]
            rim = ss(0.75, 1.0, np.hypot(X, Y))
            surf = mix(surf, rgb(0.35, 0.6, 1.0) * (0.1 + 0.9 * day)[:, None], 0.45 * rim)

            disc = ss(1.0, 0.985, np.hypot(X, Y))
            col[g] = mix(col[g], surf, disc)
            fe = np.where(land > 0.5, LAND, SEA)
            fe[cloud > 0.5] = SCLERA
            fe[city > 0.4] = SUN
            fam[g] = np.where(disc > 0.5, fe, fam[g])
            ov[g[disc > 0.5]] = -1

        # home: a warm light that opens up as we fall toward it
        px, py = self.HOME
        bd = np.hypot(x - px, y - py)
        r_in = 0.01 * (1 + 8 * e)
        blend_inner(col, fam, ov, ss(r_in * 2.5, r_in, bd) * (0.3 + 0.7 * e), inner)
        glow = ((1.3 * np.exp(-(bd / 0.012) ** 2) + 0.4 * np.exp(-bd / 0.035))
                * (0.75 + 0.25 * math.sin(fr.t * 3)) * (1 - e) ** 2)
        col += rgb(1.0, 0.72, 0.35) * glow[:, None]
        hot = glow > 0.5
        fam[hot] = SUN
        ov[hot] = -1
        return col, fam, ov


class HomeWorld:
    zoom0, cam0, zoom_end = 1.0, (0.0, 0.02), 60.0
    SIZE, BASE = 1.6, -0.22       # the house is drawn in units of SIZE, standing on y = BASE
    WINDOW = (0.0, BASE + 0.42 * SIZE)   # round window up in the gable
    WINDOW_R = 0.05 * SIZE

    def portal(self, fr, e):
        return self.WINDOW[0], self.WINDOW[1], 9.0

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        t = fr.t
        k = x.size
        col = mix(fill(k, rgb(0.08, 0.09, 0.19)), rgb(0.01, 0.015, 0.05), ss(-0.3, 0.9, y))
        fam = np.full(k, SPACE, np.int64)
        ov = np.full(k, -1, np.int64)

        # a still night sky, a crescent moon
        tw = bil(TW, x * 45 + 5, y * 45 + 9)
        star = ss(0.93, 0.99, tw) * (0.6 + 0.4 * np.sin(t * 2.5 + tw * 60)) * ss(-0.1, 0.1, y)
        col += rgb(0.9, 0.92, 1.0) * star[:, None]
        on = star > 0.15
        ov[on] = fr.sparkles(star[on] * 1.2)
        mx, my = -1.0, 0.62
        md = np.hypot(x - mx, y - my)
        moon = ss(0.105, 0.095, md) * ss(0.08, 0.095, np.hypot(x - mx - 0.045, y - my - 0.025))
        col += rgb(0.5, 0.55, 0.75) * (0.18 * np.exp(-md / 0.15))[:, None]
        col = mix(col, rgb(0.98, 0.94, 0.80), moon)
        fam[moon > 0.5] = SCLERA
        ov[moon > 0.3] = -1

        # hills
        far = ss(0.005, -0.005, y - (-0.10 + 0.07 * np.cos(x * 1.1 - 0.4) + 0.03 * np.cos(x * 2.9 + 1.0)))
        col = mix(col, rgb(0.03, 0.04, 0.09), far)
        yg = -0.27 + 0.05 * np.cos(x * 1.4 + 0.3) + 0.02 * np.cos(x * 4.1)
        ground = ss(0.005, -0.005, y - yg)
        grass = rgb(0.05, 0.11, 0.05) * (0.6 + 0.8 * bil(TN, x * 70, y * 70))[:, None]
        col = mix(col, grass, ground)
        hill = (far > 0.5) | (ground > 0.5)
        fam[hill] = LAND
        ov[hill] = -1

        # the house, in its own units (u, v): walls up to v = 0.10, roof apex at v = 0.40
        S, B = self.SIZE, self.BASE
        u, v = x / S, (y - B) / S + B
        au = np.abs(u)
        wall = (au < 0.34) & (v > B) & (v < 0.10)
        roof = (v >= 0.095) & (v < 0.40 - au * (0.30 / 0.44)) & (au < 0.44)
        chimney = (np.abs(u - 0.24) < 0.04) & (v > 0.18) & (v < 0.36)
        planks = 0.85 + 0.15 * np.cos(v * 2 * np.pi / 0.035)
        wood = rgb(0.46, 0.22, 0.11) * (planks * (0.85 + 0.3 * bil(TN, u * 50, v * 10)))[:, None]
        shingle = rgb(0.40, 0.13, 0.08) * (0.75 + 0.25 * np.cos(v * 2 * np.pi / 0.03))[:, None]
        shingle *= (1 + 0.5 * ss(0.05, 0.0, 0.40 - au * (0.30 / 0.44) - v))[:, None]   # moonlit roof edge
        col[wall] = wood[wall]
        col[roof] = shingle[roof]
        col[chimney] = rgb(0.45, 0.16, 0.10)
        house = wall | roof | chimney
        fam[house] = HOUSE
        ov[house] = -1

        lamp = rgb(1.0, 0.76, 0.38) * (0.93 + 0.07 * math.sin(t * 7) * math.sin(t * 3.1))
        glow = np.zeros(k, np.float32)
        lit = np.zeros(k, bool)
        for cu in (-0.2, 0.2):
            pane = (np.abs(u - cu) < 0.075) & (np.abs(v + 0.05) < 0.065)
            frame = pane & ((np.abs(u - cu) < 0.008) | (np.abs(v + 0.05) < 0.012))
            col[pane] = lamp
            col[frame] = rgb(0.15, 0.07, 0.04)
            lit |= pane & ~frame
            glow += 0.35 * np.exp(-np.hypot(u - cu, (v + 0.05) * 1.3) / 0.09)
        door = (au < 0.065) & (v > B) & (v < -0.02)
        col[door] = rgb(0.22, 0.09, 0.04)
        col[door & (au > 0.05)] = lamp * 0.5          # light leaking round the door
        fam[lit] = SUN
        words = wall & ~lit & ~door
        ov[words] = fr.text(cells[words], "home ")    # the walls are written in it
        col += rgb(1.0, 0.6, 0.28) * (glow * ~lit)[:, None]

        # chimney smoke
        rise = v - 0.37
        smoke_u = np.abs(u - 0.24 - 0.05 * np.sin(rise * 8 - t * 0.8) * rise * 4)
        smoke = (ss(0.02 + rise * 0.25, 0.0, smoke_u) * ss(0.0, 0.03, rise) * ss(0.35, 0.1, rise)
                 * ss(0.35, 0.7, bil(TN, u * 25, (v - t * 0.06) * 25)))
        col = mix(col, rgb(0.35, 0.35, 0.42), 0.45 * smoke)

        # the round window: warm glass that opens onto what is inside
        wx, wy = self.WINDOW
        R = self.WINDOW_R
        wd = np.hypot(x - wx, y - wy)
        glass = ss(R, R * 0.88, wd)
        col = mix(col, lamp, glass)
        fam[glass > 0.5] = SUN
        blend_inner(col, fam, ov, glass * (0.45 + 0.55 * e), inner)
        cross = ((np.abs(x - wx) < 0.012) | (np.abs(y - wy) < 0.02)) & (wd < R)
        col = mix(col, rgb(0.15, 0.07, 0.04), cross * (1 - e) ** 2)
        ring = ss(R * 0.88, R, wd) * ss(R * 1.28, R * 1.16, wd)
        col = mix(col, rgb(0.2, 0.09, 0.05), ring)
        col += rgb(1.0, 0.6, 0.28) * (0.3 * np.exp(-np.maximum(wd - R, 0) / 0.08) * (wd > R))[:, None]
        return col, fam, ov


class HeartWorld:
    zoom0, cam0, zoom_end = 1.0, (0.0, 0.0), 10.0
    CENTRE = (0.0, 0.02)

    def __init__(self, rng):
        n = 70
        self.px, self.py = rng.random(n), rng.random(n)
        self.speed = rng.uniform(0.5, 1.5, n)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.col = np.where(rng.random(n)[:, None] < 0.6, rgb(1.0, 0.45, 0.6), rgb(1.0, 0.8, 0.5))

    def portal(self, fr, e):
        return self.CENTRE[0], self.CENTRE[1], 11.0

    def shade(self, fr, x, y, sx, sy, cells, T, view, e, inner):
        t = fr.t
        ph = (t * 1.15) % 1.0                                  # lub-dub
        beat = math.exp(-(ph / 0.07) ** 2) + 0.6 * math.exp(-((ph - 0.25) / 0.07) ** 2) \
            + math.exp(-((ph - 1) / 0.07) ** 2)
        S = 0.48 * (1 + 0.05 * beat)
        hx, hy = x / S, y / S
        q = hx * hx + hy * hy - 1
        f = q * q * q - hx * hx * hy * hy * hy                 # < 0 inside the heart
        inside = ss(0.02, -0.02, f)
        depth = np.clip(-f, 0, 1) ** 0.35
        heart = mix(fill(x.size, rgb(0.45, 0.0, 0.07)), rgb(1.0, 0.22, 0.40), depth)
        shine = np.exp(-(((hx + 0.45) / 0.28) ** 2 + ((hy - 0.5) / 0.2) ** 2))
        heart = mix(heart, rgb(1.0, 0.85, 0.9), 0.6 * shine)

        bg = mix(fill(x.size, rgb(0.17, 0.02, 0.06)), rgb(0.03, 0.0, 0.01), ss(0.0, 1.3, np.hypot(x, y)))
        halo = 0.45 * np.exp(-np.clip(f, 0, None) * 4.0) * (f > 0) * (0.7 + 0.3 * beat)
        bg += rgb(0.9, 0.15, 0.3) * halo[:, None]
        col = mix(bg, heart, inside)
        body = inside > 0.5
        fam = np.where(body, HEART, SPACE)
        ov = np.full(x.size, -1, np.int64)
        ov[body] = fr.text(cells[body], "love ")

        # little sparks drifting up
        camx, camy, zoom = view
        wx = self.px * 3.8 - 1.9 + 0.05 * np.sin(t * 0.8 * self.speed + self.phase)
        wy = (self.py + t * 0.035 * self.speed) % 1.0 * 2.4 - 1.2
        w = (0.6 + 0.4 * np.sin(t * 2 * self.speed + self.phase)) * min(1.0, T[0] ** 2)
        acc = fr.splat((wx - camx) * zoom, (wy - camy) * zoom, w, self.col, T)
        b = acc[0, cells]
        col += acc[1:, cells].T
        on = (b > 0.2) & ~body
        ov[on] = fr.sparkles(b[on])

        # and at its centre, an eye again
        d = np.hypot(x - self.CENTRE[0], y - self.CENTRE[1])
        r_in = 0.05 * (1 + 5 * e)
        blend_inner(col, fam, ov, ss(r_in * 2.2, r_in, d) * ss(0.0, 0.35, e), inner)
        return col, fam, ov


# ------------------------------------------------------------------ the journey

class Journey:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.t = 0.0
        self.eye = EyeWorld(self.rng)
        self.worlds = (self.eye, GalaxyWorld(), SolarWorld(), EarthWorld(), HomeWorld(), HeartWorld(self.rng))
        self.pos = 0.0          # 2.4 = 40% of the way from world 2 into world 3
        self.target = 0
        self.auto = False
        self.rest = 0.0
        self.speed = 0.05
        self.stars = Stars(3500, self.rng)
        self.orbit_t = 0.0
        self.earth_t = 0.0
        self.eye_theme = 0
        self._shape = None

    def forward(self):
        self.target = math.floor(self.pos + 1e-9) + 1
        self.rest = 0.0

    def back(self):
        self.target = math.ceil(self.pos - 1e-9) - 1
        self.rest = 0.0

    def scrub(self, delta):
        self.auto = False
        self.rest = 0.0
        self.target = self.pos + delta

    def jump_to(self, idx):
        self.auto = False
        self.rest = 0.0
        n = len(self.worlds)
        curr = self.pos % n
        target_norm = idx % n
        diff = target_norm - curr
        if diff > n / 2:
            diff -= n
        elif diff < -n / 2:
            diff += n
        self.target = self.pos + diff

    def update(self, dt):
        self.t += dt
        n = len(self.worlds)
        if self.pos == self.target:
            self.rest += dt
            if self.auto and self.rest > HOLD_TIME * (1.5 if self.pos == 0 else 1.0):
                self.forward()
        step = dt / TRAVEL_TIME
        if self.target > self.pos:
            self.pos = min(self.pos + step, self.target)
        elif self.target < self.pos:
            self.pos = max(self.pos - step, self.target)
        if self.pos >= n:                       # through the heart and back into the eye
            self.pos -= n
            self.target -= n
        elif self.pos < 0:
            self.pos += n
            self.target += n

        base = math.floor(self.pos)
        f = self.pos - base
        e = float(ss(0.0, 1.0, f))
        self.speed = 0.05 + (1.1 * math.sin(math.pi * f) ** 2 if base <= 2 else 0.0)
        self.stars.step(dt, self.speed)
        self.orbit_t += dt * (1 - e if base == 2 else 1.0)    # hold still while we land
        self.earth_t += dt * (1 - e if base == 3 else 1.0)
        self.eye.update(dt, busy=(base == 0 and f > 0.02))

    def render(self, k, sx, sy, cells, T, f, depth, fr):
        """Shade world k (with the next worlds nested in its portal) for the given cells."""
        w = self.worlds[k % len(self.worlds)]
        e, c = float(ss(0.0, 1.0, f)), float(ss(0.0, 0.6, f))
        px, py, K0 = w.portal(fr, e)
        zoom = w.zoom0 * (w.zoom_end / w.zoom0) ** e
        camx = w.cam0[0] + (px - w.cam0[0]) * c
        camy = w.cam0[1] + (py - w.cam0[1]) * c
        x, y = camx + sx / zoom, camy + sy / zoom

        def inner(pm):
            if depth == 0:
                return None
            sel = np.flatnonzero(pm > 0.003)
            if not sel.size:
                return None
            K = math.exp(math.log(K0) * (1 - e) + math.log(zoom) * e)
            Tn = (T[0] * zoom / K, T[0] * (px - camx) * zoom + T[1], T[0] * (py - camy) * zoom + T[2])
            return (sel,) + self.render(k + 1, (x[sel] - px) * K, (y[sel] - py) * K, cells[sel], Tn, 0.0,
                                        depth - 1, fr)

        return w.shade(fr, x, y, sx, sy, cells, T, (camx, camy, zoom), e, inner)

    def _cells(self, W, H):
        if self._shape != (H, W):
            self._shape = (H, W)
            self.R0 = self.rng.integers(0, VARIANTS, (H, W))
            self.R1 = self.rng.random((H, W), dtype=np.float32)
            self.R2 = self.rng.random((H, W), dtype=np.float32)

    def _grid(self, W, H, cw, ch, half):
        key = (W, H, cw, ch, half)
        if getattr(self, "_grid_key", None) != key:
            self._grid_key = key
            gx = (((np.arange(W, dtype=np.float32) + 0.5) * cw - W * cw / 2) / half)
            gy = (-((np.arange(H, dtype=np.float32) + 0.5) * ch - H * ch / 2) / half)
            sx, sy = np.meshgrid(gx, gy)
            self._sx = np.ascontiguousarray(sx.ravel())
            self._sy = np.ascontiguousarray(sy.ravel())
            self._cells_idx = np.arange(W * H)
        return self._sx, self._sy, self._cells_idx

    def frame(self, glyphs, W, H, glow=True):
        """Shade one frame. Returns character ids (H, W) and uint8 fg/bg colours (H, W, 3)."""
        self._cells(W, H)
        fr = Frame(self, glyphs, W, H)
        cw, ch, half = glyphs.cw, glyphs.ch, fr.half
        sx, sy, cells_idx = self._grid(W, H, cw, ch, half)
        base = math.floor(self.pos)
        col, fam, ov = self.render(base, sx, sy, cells_idx, (1.0, 0.0, 0.0), self.pos - base, 2, fr)
        col, fam, ov = col.reshape(H, W, 3), fam.reshape(H, W), ov.reshape(H, W)

        vig = 1 - 0.55 * ss(0.7, 1.6, np.hypot(sx * 0.6, sy)).reshape(H, W)
        col *= (vig * (1 + 0.15 * self.eye.light))[..., None]
        np.clip(col, 0, None, out=col)

        # ---- choose characters
        L = col @ np.array([0.299, 0.587, 0.114], np.float32)
        level = np.clip(np.rint(np.clip(L, 0, 1) ** 0.9 * 1.05 * (LEVELS - 1) - 0.4), 0, LEVELS - 1).astype(np.int64)
        var = (self.R0 + np.floor(self.t * 0.4 + self.R1).astype(np.int64)) % VARIANTS
        idx = glyphs.lut[fam, level, var]

        # contours drawn as strokes that follow them
        Lp = np.pad(L, 1, mode="edge")
        Lb = (Lp[:-2, 1:-1] + Lp[2:, 1:-1] + Lp[1:-1, :-2] + Lp[1:-1, 2:] + 2 * L) / 6   # soften pores
        gxL = np.gradient(Lb, axis=1)
        gyL = -np.gradient(Lb, axis=0) * (cw / ch)
        edge = np.isin(fam, EDGE_FAMILIES) & (ov < 0) & (gxL * gxL + gyL * gyL > 0.07 ** 2) & (level > 0)
        idx[edge] = glyphs.strokes[stroke_index(np.degrees(np.arctan2(gyL[edge], gxL[edge])) + 90)]
        idx = np.where(ov >= 0, ov, idx)

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


def draw(journey, glyphs, W, H, glow=True):
    return glyphs.compose(*journey.frame(glyphs, W, H, glow))


# ------------------------------------------------------------------ app

EYE_THEMES = (
    ("Natural Human (Blue)", "Lifelike skin & ocean blue-hazel eye"),
    ("Natural Human (Emerald)", "Lifelike skin & emerald green-hazel eye"),
    ("Original (Bronze & Amber)", "Original deep bronze skin & amber eye"),
)

HELP = ("mouse: look/scrub   space/back: travel   1-6: worlds   A: auto   "
        "P: pause   C: eye style   M: sound   B: blink   +/-: size   G: glow   S: save   H: help   Esc: quit")


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    pygame.init()
    pygame.display.set_caption("In Your Eyes")
    windowed = (1280, 760)
    screen = pygame.display.set_mode(windowed, pygame.RESIZABLE)
    font_px = 13
    glyphs = Glyphs(font_px)
    hud = pygame.font.SysFont("consolas,couriernew", 14)
    journey = Journey()
    eye = journey.eye
    audio = SoundEngine()
    clock = pygame.time.Clock()
    glow, help_until, fullscreen = True, 8.0, False
    paused = False
    toast_text = ""
    toast_until = 0.0

    while True:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        sw, sh = screen.get_size()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.MOUSEWHEEL:
                journey.scrub(ev.y * 0.22)
                toast_text, toast_until = f"Scrubbing (depth: {journey.pos:.1f})", time.time() + 1.2
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    return
                elif k == pygame.K_SPACE:
                    journey.forward()
                elif k == pygame.K_BACKSPACE:
                    journey.back()
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    w_idx = k - pygame.K_1
                    names = ("Eye", "Galaxy", "Solar System", "Earth", "Home", "Heart")
                    journey.jump_to(w_idx)
                    toast_text, toast_until = f"World: {names[w_idx]}", time.time() + 1.8
                elif k == pygame.K_a:
                    journey.auto = not journey.auto
                    journey.rest = 0.0
                    toast_text, toast_until = "Autoplay: ON" if journey.auto else "Autoplay: OFF", time.time() + 1.5
                elif k == pygame.K_p:
                    paused = not paused
                    toast_text, toast_until = "PAUSED" if paused else "RESUMED", time.time() + 1.5
                elif k == pygame.K_c:
                    journey.eye_theme = (journey.eye_theme + 1) % len(EYE_THEMES)
                    toast_text, toast_until = f"Eye: {EYE_THEMES[journey.eye_theme][0]}", time.time() + 1.8
                elif k == pygame.K_m:
                    muted = audio.toggle_mute()
                    toast_text, toast_until = "Sound: MUTED" if muted else "Sound: ON", time.time() + 1.5
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
                    toast_text, toast_until = "Glow: ON" if glow else "Glow: OFF", time.time() + 1.2
                elif k == pygame.K_h:
                    help_until = 0.0 if journey.t < help_until else journey.t + 1e9
                elif k == pygame.K_s:
                    name = time.strftime("eye_%Y%m%d_%H%M%S.png")
                    pygame.image.save(screen, name)
                    toast_text, toast_until = f"Saved {name}", time.time() + 2.0
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
                eye.look_at(EyeWorld.cam0[0] + (ev.pos[0] - sw / 2) / (sh / 2) / EyeWorld.zoom0,
                            EyeWorld.cam0[1] - (ev.pos[1] - sh / 2) / (sh / 2) / EyeWorld.zoom0)

        if not paused:
            journey.update(dt)
        audio.update(journey.pos, 0.0 if paused else journey.speed, dt)

        sw, sh = screen.get_size()
        W, H = max(sw // glyphs.cw, 1), max(sh // glyphs.ch, 1)
        screen.fill((0, 0, 0))
        screen.blit(draw(journey, glyphs, W, H, glow), ((sw - W * glyphs.cw) // 2, (sh - H * glyphs.ch) // 2))

        now = time.time()
        if now < toast_until and toast_text:
            t_alpha = int(255 * min(1.0, (toast_until - now) / 0.4))
            t_surf = hud.render(f"  {toast_text}  ", True, (240, 240, 240), (25, 25, 25))
            t_surf.set_alpha(t_alpha)
            screen.blit(t_surf, (sw - t_surf.get_width() - 16, 16))

        if journey.t < help_until:
            text = hud.render(HELP, True, (150, 150, 150), (0, 0, 0))
            text.set_alpha(int(255 * min(1.0, help_until - journey.t)))
            screen.blit(text, (12, sh - text.get_height() - 10))
        pygame.display.flip()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:     # Ctrl+C in the terminal: just close quietly
        pass
    finally:
        pygame.quit()
