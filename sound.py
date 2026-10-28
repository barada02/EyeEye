"""Procedural audio engine for 'In Your Eyes'.
Generates real-time ambient soundscapes, resonant drones, heartbeat pulses,
and warp whoosh using pure NumPy waveform synthesis and pygame.mixer.
Requires zero external audio files.
"""

import math
import numpy as np
import pygame

SAMPLE_RATE = 44100
DUR = 4.0   # seconds per looping buffer


def _smooth_loop(arr, fade_len=512):
    """Ensure seamless looping by cross-fading the start and end."""
    w = np.linspace(0, 1, fade_len, dtype=np.float32)
    arr[:fade_len] = arr[:fade_len] * w[:, None] + arr[-fade_len:] * (1 - w[:, None])
    return arr


def _to_stereo_sound(left, right, gain=0.35):
    """Normalize and convert two float arrays to a 16-bit stereo pygame Sound."""
    stereo = np.column_stack([left, right])
    peak = np.max(np.abs(stereo)) + 1e-6
    if peak > 1.0:
        stereo = stereo / peak
    stereo = (stereo * (gain * 32767)).astype(np.int16)
    _smooth_loop(stereo)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def _gen_pink_noise(n, rng):
    """Generate simple filtered pink/warm noise."""
    white = rng.standard_normal(n).astype(np.float32)
    # 3-pole IIR approximation for pink noise
    b0 = b1 = b2 = 0.0
    out = np.empty(n, np.float32)
    for i in range(n):
        w = white[i]
        b0 = 0.99765 * b0 + w * 0.0990460
        b1 = 0.96300 * b1 + w * 0.2965164
        b2 = 0.57000 * b2 + w * 1.0526913
        out[i] = b0 + b1 + b2 + w * 0.1848
    return out / (np.max(np.abs(out)) + 1e-5)


class SoundEngine:
    """Multi-channel procedural ambient sound generator."""

    def __init__(self):
        self.enabled = False
        self.muted = False
        self.channels = []
        self.sounds = []
        self.whoosh_channel = None
        self.whoosh_sound = None
        self.master_vol = 0.8
        self.current_vols = [0.0] * 6
        self.whoosh_vol = 0.0

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(16)
            self._synthesize()
            self.enabled = True
        except Exception as e:
            # If sound hardware is unavailable, run silently
            print(f"[Audio] Sound disabled: {e}")

    def _synthesize(self):
        """Synthesize all 6 world ambient loops and warp sound."""
        n = int(SAMPLE_RATE * DUR)
        t = np.linspace(0, DUR, n, endpoint=False, dtype=np.float32)
        rng = np.random.default_rng(42)

        # ---------------- 0: Eye (Mystical room tone & deep binaural drone)
        # Fundamental C2 (65.4 Hz) + G2 (98 Hz) with slow 0.25 Hz LFO modulation
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
        s_eye_l = (np.sin(2 * np.pi * 65.4 * t) * 0.6 +
                   np.sin(2 * np.pi * 98.0 * t) * 0.3 +
                   np.sin(2 * np.pi * 130.8 * t) * 0.15) * lfo
        s_eye_r = (np.sin(2 * np.pi * 65.8 * t) * 0.6 +  # slight detuning for binaural beat
                   np.sin(2 * np.pi * 98.2 * t) * 0.3 +
                   np.sin(2 * np.pi * 131.0 * t) * 0.15) * lfo
        snd_eye = _to_stereo_sound(s_eye_l, s_eye_r, gain=0.45)

        # ---------------- 1: Galaxy (Deep cosmic space pad)
        # Low A1 (55 Hz) + A2 (110 Hz) + E3 (165 Hz) + soft pink noise background
        pink1 = _gen_pink_noise(n, rng) * 0.12
        pink2 = _gen_pink_noise(n, rng) * 0.12
        s_gal_l = np.sin(2 * np.pi * 55.0 * t) * 0.5 + np.sin(2 * np.pi * 110.0 * t) * 0.3 + np.sin(2 * np.pi * 165.0 * t) * 0.2 + pink1
        s_gal_r = np.sin(2 * np.pi * 55.2 * t) * 0.5 + np.sin(2 * np.pi * 110.4 * t) * 0.3 + np.sin(2 * np.pi * 165.2 * t) * 0.2 + pink2
        snd_galaxy = _to_stereo_sound(s_gal_l, s_gal_r, gain=0.40)

        # ---------------- 2: Solar System (Celestial chime harmonics)
        # D major overtones: 146.8 Hz, 220.0 Hz, 293.6 Hz, 440.0 Hz with phase shifts
        s_sol_l = (np.sin(2 * np.pi * 146.8 * t) * 0.35 +
                   np.sin(2 * np.pi * 220.0 * t) * 0.30 +
                   np.sin(2 * np.pi * 293.6 * t) * 0.20 +
                   np.sin(2 * np.pi * 440.0 * t) * 0.15)
        s_sol_r = (np.sin(2 * np.pi * 147.0 * t) * 0.35 +
                   np.sin(2 * np.pi * 220.3 * t) * 0.30 +
                   np.sin(2 * np.pi * 294.0 * t) * 0.20 +
                   np.sin(2 * np.pi * 439.5 * t) * 0.15)
        snd_solar = _to_stereo_sound(s_sol_l, s_sol_r, gain=0.35)

        # ---------------- 3: Earth (Atmospheric wind / ocean surf wash)
        # Modulated pink noise with sweeping low-pass filter
        surf_env = (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)) ** 2
        p_earth = _gen_pink_noise(n, rng)
        s_earth_l = (p_earth * (0.3 + 0.7 * surf_env) + np.sin(2 * np.pi * 82.4 * t) * 0.3)
        s_earth_r = (np.roll(p_earth, 12000) * (0.3 + 0.7 * np.roll(surf_env, 8000)) + np.sin(2 * np.pi * 82.7 * t) * 0.3)
        snd_earth = _to_stereo_sound(s_earth_l, s_earth_r, gain=0.38)

        # ---------------- 4: Home (Warm acoustic chord & fireplace glow)
        # Warm C major 7 chord: C3 (130.8), E3 (164.8), G3 (196.0), B3 (246.9)
        crackle = np.where(rng.random(n) > 0.9985, rng.uniform(-0.4, 0.4, n), 0.0)
        s_home_l = (np.sin(2 * np.pi * 130.8 * t) * 0.35 +
                    np.sin(2 * np.pi * 164.8 * t) * 0.25 +
                    np.sin(2 * np.pi * 196.0 * t) * 0.20 +
                    np.sin(2 * np.pi * 246.9 * t) * 0.15 + crackle)
        s_home_r = (np.sin(2 * np.pi * 131.0 * t) * 0.35 +
                    np.sin(2 * np.pi * 165.0 * t) * 0.25 +
                    np.sin(2 * np.pi * 196.2 * t) * 0.20 +
                    np.sin(2 * np.pi * 247.1 * t) * 0.15 + np.roll(crackle, 4000))
        snd_home = _to_stereo_sound(s_home_l, s_home_r, gain=0.38)

        # ---------------- 5: Heart (Synchronized rhythmic lub-dub heartbeat)
        # Visual heart rate is approx 1.15 Hz -> loop of 2 beats per ~1.74s
        hb = np.zeros(n, np.float32)
        beat_period = int(SAMPLE_RATE / 1.15)
        for start_idx in range(0, n - beat_period // 2, beat_period):
            # Lub (first systolic sound ~55Hz)
            m1 = min(int(SAMPLE_RATE * 0.14), n - start_idx)
            t1 = np.arange(m1, dtype=np.float32) / SAMPLE_RATE
            env1 = np.sin(np.pi * t1 / 0.14) ** 2
            hb[start_idx:start_idx + m1] += np.sin(2 * np.pi * 55.0 * t1) * env1 * 0.8

            # Dub (second diastolic sound ~75Hz, ~0.22s later)
            d_offset = start_idx + int(SAMPLE_RATE * 0.22)
            if d_offset < n:
                m2 = min(int(SAMPLE_RATE * 0.12), n - d_offset)
                t2 = np.arange(m2, dtype=np.float32) / SAMPLE_RATE
                env2 = np.sin(np.pi * t2 / 0.12) ** 2
                hb[d_offset:d_offset + m2] += np.sin(2 * np.pi * 75.0 * t2) * env2 * 0.55

        hb_warm = hb + np.sin(2 * np.pi * 45.0 * t) * 0.15
        snd_heart = _to_stereo_sound(hb_warm, hb_warm, gain=0.55)

        # ---------------- Warp Whoosh (dynamic rushing noise)
        whoosh_noise = _gen_pink_noise(n, rng)
        w_env = (0.5 + 0.5 * np.sin(2 * np.pi * (1.0 / DUR) * t))
        whoosh_l = whoosh_noise * w_env
        whoosh_r = np.roll(whoosh_noise, 4000) * w_env
        snd_whoosh = _to_stereo_sound(whoosh_l, whoosh_r, gain=0.60)

        # Allocate channels and start loops
        self.sounds = [snd_eye, snd_galaxy, snd_solar, snd_earth, snd_home, snd_heart]
        self.channels = [pygame.mixer.Channel(i) for i in range(6)]
        for i, (ch, snd) in enumerate(zip(self.channels, self.sounds)):
            ch.play(snd, loops=-1)
            ch.set_volume(0.0)

        self.whoosh_sound = snd_whoosh
        self.whoosh_channel = pygame.mixer.Channel(6)
        self.whoosh_channel.play(snd_whoosh, loops=-1)
        self.whoosh_channel.set_volume(0.0)

    def toggle_mute(self):
        """Toggle mute state. Returns new muted boolean."""
        self.muted = not self.muted
        if self.muted:
            for ch in self.channels:
                ch.set_volume(0.0)
            if self.whoosh_channel:
                self.whoosh_channel.set_volume(0.0)
        return self.muted

    def update(self, pos, speed, dt):
        """Smoothly crossfade channels according to current journey position and speed."""
        if not self.enabled or self.muted:
            return

        n_worlds = 6
        norm_pos = pos % n_worlds

        # Smooth exponential smoothing factor
        alpha = min(1.0, dt * 8.0)

        for i, ch in enumerate(self.channels):
            # Circular distance between current position and world i
            diff = abs(norm_pos - i)
            d = min(diff, n_worlds - diff)
            target_vol = max(0.0, 1.0 - d * 1.1) ** 1.8 * self.master_vol
            self.current_vols[i] += (target_vol - self.current_vols[i]) * alpha
            ch.set_volume(self.current_vols[i])

        # Warp whoosh volume scales with travel speed
        target_whoosh = np.clip((speed - 0.08) / 0.85, 0.0, 1.0) * (self.master_vol * 0.7)
        self.whoosh_vol += (target_whoosh - self.whoosh_vol) * alpha
        if self.whoosh_channel:
            self.whoosh_channel.set_volume(self.whoosh_vol)
