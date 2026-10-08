#!/usr/bin/env python3
"""
SOLAR SYSTEM GENESIS  -  interactive N-body simulation (pygame + numpy)
=======================================================================
Watch a rotating nebula collapse, flatten into a protoplanetary disk, and
accrete into planets while a young star ignites and blows the gas away.
Top-down view. Everything is real N-body gravity (velocity-Verlet), with gas
drag, inelastic accretion (merging with momentum conservation) and a
growing central protostar.

Install :  pip install pygame numpy
Run     :  python solar_system_genesis.py

MOUSE
  Left click (empty space)   spawn a body on a circular orbit (type: keys 1-4)
  Left drag                  slingshot launch, predicted path is shown
  Shift + left click         burst of dust on circular orbits
  Left click on a body       select + follow it (info panel)
  Right button (hold)        gravity attractor at the cursor  (Shift = repeller)
  Mouse wheel                zoom (toward cursor once you have panned)
  Middle drag / Ctrl + drag  pan camera

KEYS
  SPACE pause | UP/DOWN speed | [ ] accretion radius | 1-4 spawn type
  T trails | L frost line | G nebula glow | A auto-zoom | F follow star
  D add dust | R restart | S screenshot | H help | ESC quit
"""
import math
import sys
import json

import numpy as np
import pygame

# --------------------------------------------------------------------------
# Physics / scenario constants (simulation units, G = 1)
# --------------------------------------------------------------------------
G = 1.0
EPS = 4.0                    # gravitational softening
DT = 0.1                     # time step
STAR_M0 = 1.0                # begin with a tiny protostellar seed
STAR_M1 = 120.0              # stable star ignition scale
BH_THRESHOLD = 3500.0        # deliberately high simulation-unit collapse threshold
STAR_FUEL_RATE = 0.00002    # fuel consumption; mass-dependent
N_START = 320
MAX_BODIES = 520
R_IN, R_OUT = 220.0, 900.0   # initial cloud extent
V_FRAC = (0.45, 0.80)        # initial speed as fraction of circular (sub-orbital -> infall)
FROST = 175.0                # frost line radius (icy bodies beyond)
GAMMA = 0.02                 # gas drag strength
MOUSE_GM = 5000.0            # strength of the mouse attractor
LAUNCH_K = 0.05              # slingshot velocity per screen pixel (at zoom 1)
TRAIL_LEN = 40
SAVE_FILE = "my_solar_system.json"

T_COLLAPSE = 350.0           # end of stage 1
T_DISPERSE = 1300.0          # gas starts to clear
T_CLEAR = 2100.0             # gas gone
T_TO_MYR = 1.0 / 21.0        # cosmetic time conversion

STAGES = [
    (0.0, T_COLLAPSE, "1  Star formation / cloud collapse"),
    (T_COLLAPSE, T_DISPERSE, "2  Protoplanetary disk / accretion"),
    (T_DISPERSE, T_CLEAR, "3  Stellar wind clears the gas"),
    (T_CLEAR, float("inf"), "4  Mature planetary system"),
]

SPAWN = {1: ("dust", 1.0), 2: ("planetesimal", 8.0),
         3: ("protoplanet", 40.0), 4: ("giant core", 180.0)}

# colours per mass class [rocky, icy]
COL = {
    0: ((165, 155, 145), (175, 190, 205)),
    1: ((200, 175, 140), (195, 215, 235)),
    2: ((215, 140, 95), (140, 190, 235)),
    3: ((85, 160, 235), (235, 190, 125)),
    4: ((100, 200, 150), (240, 205, 150)),
}


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def stage_of(t):
    for k, (a, b, name) in enumerate(STAGES):
        if a <= t < b:
            return k, name
    return 3, STAGES[3][2]


def classify(m, icy):
    """Return (class_index, human readable type)."""
    if m < 2.5:
        return 0, "Dust cloud"
    if m < 15:
        return 1, "Planetesimal"
    if m < 60:
        return 2, "Protoplanet"
    if icy:
        return (4, "Gas giant") if m >= 120 else (3, "Ice giant")
    return (4, "Super-Earth") if m >= 150 else (3, "Rocky planet")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
class Sim:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # TRUE ZERO-START SANDBOX: only a tiny numerical protostellar seed
        # exists so that the N-body solver has a gravitational center.
        # No planets, asteroids or finished Sun are created automatically.
        self.pos = np.array([[0.0, 0.0]], dtype=float)
        self.vel = np.array([[0.0, 0.0]], dtype=float)
        self.m = np.array([STAR_M0], dtype=float)
        self.ids = np.array([0], dtype=int)
        self.next_id = 1
        self.names = {0: "Sun"}
        self.custom_icy = {}
        for ident in self.ids[1:]:
            self.names[int(ident)] = f"Body {int(ident)}"
        self.created = 0
        self.star_fuel = 1.0
        self.star_stage = "protostar"
        self.black_hole = False
        self.collapse_reason = ""
        self.hist = np.repeat((self.pos - self.pos[0])[:, None, :], TRAIL_LEN, axis=1)
        self.t = 0.0
        self.absorbed = 0.0
        self.cap = 3.5            # accretion radius factor
        self.mouse = None         # (x, y, sign)
        self.a = None
        self.d2 = None
        self.remap = {}           # id merged -> id survivor (for selection)
        self.names = {0: "Sun"}
        self.custom_icy = {}      # id -> None/True/False; None = automatic frost-line classification
        self.created = 0

    # ---- helpers --------------------------------------------------------
    def gas(self):
        g = 0.12 + 0.88 * min(1.0, self.t / T_COLLAPSE)
        if self.t > T_DISPERSE:
            g *= max(0.0, 1.0 - (self.t - T_DISPERSE) / (T_CLEAR - T_DISPERSE))
        return g

    def gas_visual(self):
        if self.t <= T_DISPERSE:
            return 1.0
        return max(0.0, 1.0 - (self.t - T_DISPERSE) / (T_CLEAR - T_DISPERSE))

    def star_ramp_mass(self):
        # In the new zero-start mode the star does NOT magically gain mass.
        # Its mass changes only when gas/body material is accreted.
        return float(self.m[0])

    @staticmethod
    def star_radius(M):
        return 0.8 * max(M, 0.05) ** (1.0 / 3.0)

    @staticmethod
    def schwarzschild_radius(M):
        # Scaled Schwarzschild radius for this educational simulation.
        return max(0.08, 0.002 * M)

    def stellar_state(self):
        if self.black_hole:
            return "BLACK HOLE"
        if self.m[0] < 8:
            return "protostar"
        if self.m[0] < STAR_M1:
            return "stellar ignition"
        return "main sequence star"

    def collapse_to_black_hole(self, reason="gravitational collapse"):
        if self.black_hole:
            return
        self.black_hole = True
        self.collapse_reason = reason
        self.star_stage = "black hole"
        # Collapse conserves mass in this simplified model; radius changes
        # dramatically and the visual switches to an event horizon.
        self.a = None

    def stellar_evolution(self, dt):
        if self.black_hole:
            return
        M = self.m[0]
        self.star_fuel = max(0.0, self.star_fuel - STAR_FUEL_RATE * (max(M, 1.0) / 100.0) ** 1.7 * dt)
        if M >= BH_THRESHOLD and self.star_fuel <= 0.08:
            self.collapse_to_black_hole("unstable massive star exhausted its fuel")
        elif M >= STAR_M1:
            self.star_stage = "main sequence"
        else:
            self.star_stage = "protostar" if M < 8 else "ignition"

    def capture_radius(self):
        R = self.cap * self.m ** (1.0 / 3.0)
        R[0] = self.schwarzschild_radius(self.m[0]) * 2.5 if self.black_hole else self.star_radius(self.m[0])
        return R

    def index_of(self, ident):
        w = np.nonzero(self.ids == ident)[0]
        return int(w[0]) if len(w) else None

    # ---- dynamics -------------------------------------------------------
    def accel(self):
        pos, m = self.pos, self.m
        d = pos[None, :, :] - pos[:, None, :]          # d[i, j] = pos[j] - pos[i]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        self.d2 = r2
        s = r2 + EPS * EPS
        inv = 1.0 / (s * np.sqrt(s))
        a = G * np.einsum("ij,ijk->ik", inv * m[None, :], d)
        if self.mouse is not None:
            mx, my, sign = self.mouse
            dm = np.array([mx, my]) - pos
            rm2 = np.einsum("ij,ij->i", dm, dm) + 30.0 ** 2
            am = sign * MOUSE_GM * dm * (rm2 ** -1.5)[:, None]
            am[0] = 0.0                                  # the star ignores the mouse
            a += am
        return a

    def apply_drag(self, dt):
        g = self.gas()
        if g < 1e-3 or len(self.m) < 2:
            return
        rel = self.pos[1:] - self.pos[0]
        r = np.sqrt(np.einsum("ij,ij->i", rel, rel)) + 1e-9
        vc = np.sqrt(G * self.m[0] / np.maximum(r, 15.0))
        tang = np.stack([-rel[:, 1], rel[:, 0]], 1) / r[:, None]
        target = self.vel[0] + tang * vc[:, None]
        gamma = GAMMA * g / (1.0 + self.m[1:] / 25.0)     # big bodies feel less drag
        k = 1.0 - np.exp(-gamma * dt)
        self.vel[1:] -= (self.vel[1:] - target) * k[:, None]

    def step(self, dt=DT):
        if self.a is None:
            self.a = self.accel()
        self.vel += 0.5 * dt * self.a
        self.pos += dt * self.vel
        self.t += dt
        self.stellar_evolution(dt)
        self.a = self.accel()
        self.vel += 0.5 * dt * self.a
        self.apply_drag(dt)
        self.collide()
        self.cull()

    def collide(self):
        n = len(self.m)
        if n < 2:
            return
        R = self.capture_radius()
        if self.black_hole:
            R[0] = max(R[0], self.schwarzschild_radius(self.m[0]) * 2.5)
        thr = (R[:, None] + R[None, :]) ** 2
        mask = np.triu(self.d2 < thr, 1)
        I, J = np.nonzero(mask)
        if len(I) == 0:
            return
        dead = np.zeros(n, bool)
        for i, j in zip(I, J):
            if dead[i] or dead[j]:
                continue
            mi, mj = self.m[i], self.m[j]
            M = mi + mj
            self.pos[i] = (self.pos[i] * mi + self.pos[j] * mj) / M
            self.vel[i] = (self.vel[i] * mi + self.vel[j] * mj) / M
            if i == 0:
                self.absorbed += mj
                self.pos[0] = self.pos[0]  # star keeps its place (mass dominates)
            self.m[i] = M
            self.remap[int(self.ids[j])] = int(self.ids[i])
            dead[j] = True
        self._keep(~dead)

    def cull(self):
        rel = self.pos - self.pos[0]
        far = np.einsum("ij,ij->i", rel, rel) > 6000.0 ** 2
        far[0] = False
        if far.any():
            self._keep(~far)

    def _keep(self, keep):
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.m = self.m[keep]
        self.ids = self.ids[keep]
        self.hist = self.hist[keep]
        self.a = None

    def record_trails(self):
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = self.pos - self.pos[0]

    # ---- spawning -------------------------------------------------------
    def add_body(self, pos, vel, m, name=None, icy_override=None):
        if len(self.m) >= MAX_BODIES:
            return False
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, vel])
        self.m = np.append(self.m, m)
        self.ids = np.append(self.ids, self.next_id)
        new_id = int(self.ids[-1])
        self.next_id += 1
        self.names[new_id] = name or f"Body {new_id}"
        if icy_override is not None:
            self.custom_icy[new_id] = bool(icy_override)
        self.created += 1
        self.hist = np.concatenate(
            [self.hist, np.repeat((pos - self.pos[0])[None, None, :], TRAIL_LEN, axis=1)])
        self.a = None
        return True

    def circular_velocity(self, pos):
        rel = np.asarray(pos, float) - self.pos[0]
        r = max(float(np.hypot(*rel)), 15.0)
        v = math.sqrt(G * self.m[0] / r)
        return self.vel[0] + np.array([-rel[1], rel[0]]) / max(np.hypot(*rel), 1e-9) * v

    def add_circular(self, pos, m):
        rel = np.asarray(pos) - self.pos[0]
        if np.hypot(*rel) < self.star_radius(self.m[0]) * 1.6:
            return False
        return self.add_body(pos, self.circular_velocity(pos), m)

    def add_dust_ring(self, n=60):
        rng = self.rng
        for _ in range(n):
            r = 70 + 330 * rng.random() ** 1.3
            th = rng.random() * math.tau
            p = self.pos[0] + np.array([r * math.cos(th), r * math.sin(th)])
            v = self.circular_velocity(p) * (1 + 0.03 * rng.standard_normal())
            self.add_body(p, v, rng.uniform(0.6, 1.6))

    def add_gas_cloud(self, n=90, total_mass=500.0):
        """Create the raw material from which the player can build the star."""
        rng = self.rng
        per = total_mass / max(n, 1)
        for _ in range(n):
            r = rng.uniform(90, 700) * (0.55 + 0.45 * rng.random())
            th = rng.random() * math.tau
            p = self.pos[0] + np.array([r * math.cos(th), r * math.sin(th)])
            # Slightly sub-circular velocity encourages collapse/accretion.
            vc = math.sqrt(G * max(self.m[0], 1.0) / max(r, 15.0))
            tang = np.array([-math.sin(th), math.cos(th)])
            radial = np.array([math.cos(th), math.sin(th)])
            v = self.vel[0] + tang * vc * rng.uniform(0.15, 0.55) + radial * vc * rng.normal(0, 0.10)
            self.add_body(p, v, max(0.05, per * rng.uniform(0.25, 1.75)), name="Gas clump")

    def add_star_mass(self, amount):
        if self.black_hole:
            return
        self.m[0] += max(0.0, float(amount))
        self.absorbed += max(0.0, float(amount))
        self.stellar_evolution(0.0)
        self.a = None

    # ---- creator / sandbox helpers -------------------------------------
    def body_name(self, i):
        return self.names.get(int(self.ids[i]), f"Body {int(self.ids[i])}")

    def set_name(self, i, name):
        if i is not None and 0 <= i < len(self.ids):
            self.names[int(self.ids[i])] = name[:24] or f"Body {int(self.ids[i])}"

    def change_mass(self, i, factor):
        if i is None or i <= 0 or i >= len(self.m):
            return
        self.m[i] = float(np.clip(self.m[i] * factor, 0.05, 50000.0))
        self.a = None

    def set_circular(self, i, retro=False):
        if i is None or i <= 0 or i >= len(self.m):
            return
        self.vel[i] = self.circular_velocity(self.pos[i])
        if retro:
            self.vel[i] = 2 * self.vel[0] - self.vel[i]
        self.a = None

    def add_moon(self, parent_i, mass=0.8):
        if parent_i is None or parent_i <= 0 or parent_i >= len(self.m) or len(self.m) >= MAX_BODIES:
            return False
        parent = self.pos[parent_i].copy()
        rel = self.pos[parent_i] - self.pos[0]
        pr = max(float(np.hypot(*rel)), 20.0)
        # Hill-radius-inspired starting distance; intentionally conservative.
        hill = pr * (self.m[parent_i] / max(3 * self.m[0], 1e-9)) ** (1 / 3)
        rr = max(4.0, min(hill * 0.25, 18.0 + 3.0 * self.m[parent_i] ** (1/3)))
        ang = math.atan2(rel[1], rel[0]) + 0.7
        p = parent + rr * np.array([math.cos(ang), math.sin(ang)])
        tang = np.array([-math.sin(ang), math.cos(ang)])
        vmoon = math.sqrt(G * self.m[parent_i] / max(rr, 1e-6))
        v = self.vel[parent_i] + tang * vmoon
        return self.add_body(p, v, mass, name=f"Moon {self.created + 1}")

    def add_belt(self, n=50, r0=280.0, r1=430.0):
        rng = self.rng
        for _ in range(n):
            r = rng.uniform(r0, r1)
            th = rng.random() * math.tau
            p = self.pos[0] + np.array([r * math.cos(th), r * math.sin(th)])
            v = self.circular_velocity(p) * (1 + 0.015 * rng.standard_normal())
            self.add_body(p, v, rng.uniform(0.2, 0.8), name="Asteroid")

    def delete_body(self, i):
        if i is None or i <= 0 or i >= len(self.m):
            return False
        ident = int(self.ids[i])
        self.names.pop(ident, None)
        self.custom_icy.pop(ident, None)
        keep = np.ones(len(self.m), dtype=bool)
        keep[i] = False
        self._keep(keep)
        return True

    def save(self, filename=SAVE_FILE):
        bodies = []
        for i in range(len(self.m)):
            ident = int(self.ids[i])
            bodies.append({
                "id": ident, "name": self.names.get(ident, f"Body {ident}"),
                "mass": float(self.m[i]), "pos": self.pos[i].tolist(),
                "vel": self.vel[i].tolist(),
                "icy": self.custom_icy.get(ident, None)
            })
        data = {"version": 3, "time": float(self.t), "cap": float(self.cap),
                "next_id": int(self.next_id), "star_fuel": float(self.star_fuel),
                "black_hole": bool(self.black_hole), "star_stage": self.star_stage,
                "collapse_reason": self.collapse_reason, "bodies": bodies}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filename=SAVE_FILE):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        bodies = data["bodies"]
        self.pos = np.array([b["pos"] for b in bodies], dtype=float)
        self.vel = np.array([b["vel"] for b in bodies], dtype=float)
        self.m = np.array([b["mass"] for b in bodies], dtype=float)
        self.ids = np.array([b.get("id", i) for i, b in enumerate(bodies)], dtype=int)
        self.next_id = int(data.get("next_id", max(self.ids) + 1))
        self.names = {int(b.get("id", i)): b.get("name", f"Body {i}") for i, b in enumerate(bodies)}
        self.custom_icy = {int(b.get("id", i)): b.get("icy") for i, b in enumerate(bodies) if b.get("icy") is not None}
        self.t = float(data.get("time", 0.0))
        self.cap = float(data.get("cap", 3.5))
        self.star_fuel = float(data.get("star_fuel", 1.0))
        self.black_hole = bool(data.get("black_hole", False))
        self.star_stage = data.get("star_stage", "protostar")
        self.collapse_reason = data.get("collapse_reason", "")
        self.absorbed = 0.0
        self.hist = np.repeat((self.pos - self.pos[0])[:, None, :], TRAIL_LEN, axis=1)
        self.a = None
        self.d2 = None
        self.remap = {}

    # ---- diagnostics ----------------------------------------------------
    def orbit_info(self, i):
        mu = G * self.m[0]
        rel = self.pos[i] - self.pos[0]
        v = self.vel[i] - self.vel[0]
        r = float(np.hypot(*rel))
        sp = float(np.hypot(*v))
        E = 0.5 * sp * sp - mu / max(r, 1e-9)
        h = rel[0] * v[1] - rel[1] * v[0]
        info = {"r": r, "v": sp}
        if E < 0:
            a = -mu / (2 * E)
            e = math.sqrt(max(0.0, 1 + 2 * E * h * h / (mu * mu)))
            info.update(a=a, e=e, T=math.tau * math.sqrt(a ** 3 / mu))
        return info


# --------------------------------------------------------------------------
# Procedural sprites
# --------------------------------------------------------------------------
def make_glow(size, power=2.4):
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(x * x + y * y)
    alpha = 255 * np.clip(1 - r, 0, 1) ** power
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    rgb = np.full((size, size, 3), 255, np.uint8)
    pygame.surfarray.blit_array(surf, rgb)
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = alpha.T.astype(np.uint8)
    del pa
    return surf


def make_nebula(size=384, seed=5):
    rng = np.random.default_rng(seed)
    small = pygame.Surface((18, 18))
    pygame.surfarray.blit_array(small, (rng.random((18, 18, 3)) * 255).astype(np.uint8))
    n = pygame.surfarray.array3d(pygame.transform.smoothscale(small, (size, size)))[:, :, 0] / 255.0
    n = n.T
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    swirl = 0.72 + 0.28 * np.cos(2 * th - 7 * r)
    base = np.clip(1 - r, 0, 1) ** 1.5
    core = np.exp(-(r / 0.22) ** 2)
    alpha = np.clip((base * (0.5 + 0.9 * n) * swirl + 0.55 * core) * 0.85, 0, 1) * 255
    t = np.clip(r, 0, 1)[..., None]
    inner = np.array([255, 185, 110])
    outer = np.array([95, 110, 235])
    rgb = (inner * (1 - t) + outer * t).astype(np.uint8)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, rgb.swapaxes(0, 1))
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = alpha.T.astype(np.uint8)
    del pa
    return surf


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class App:
    def __init__(self, size=(1280, 800)):
        pygame.init()
        pygame.display.set_caption("Solar System Genesis")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        names = "consolas,menlo,dejavusansmono,couriernew,monospace"
        self.font = pygame.font.SysFont(names, 15)
        self.small = pygame.font.SysFont(names, 13)
        self.big = pygame.font.SysFont(names, 20, bold=True)

        self.sim = Sim()
        self.zoom = 0.6
        self.cam = np.array([0.0, 0.0])
        self.mode = "star"           # star | body | free
        self.autozoom = True
        self.selected = None         # body id
        self.show_trails = True
        self.show_frost = False
        self.show_nebula = True
        self.show_help = True
        self.paused = False
        self.substeps = 3
        self.spawn_key = 2
        self.frame = 0
        self.neb_r = 500.0
        self.notice = ("", 0)
        self.creator = True
        self.naming = False
        self.name_buffer = ""
        self.time_scale = 1.0

        self.press = None
        self.dragging = False
        self.panning = False
        self.attract = False

        self.glow_base = make_glow(256)
        self.glow_small = make_glow(64, 2.0)
        self.nebula = make_nebula()
        self.tint_cache, self.scale_cache = {}, {}
        self.neb_cache, self.body_glow = {}, {}
        self.bg, self.bg_size = None, None

    # ---- coordinates ----------------------------------------------------
    def center(self):
        w, h = self.screen.get_size()
        return np.array([w / 2.0, h / 2.0])

    def w2s(self, p):
        return (np.asarray(p) - self.cam) * self.zoom + self.center()

    def s2w(self, s):
        return (np.asarray(s, float) - self.center()) / self.zoom + self.cam

    def say(self, text):
        self.notice = (text, 150)

    # ---- sprites --------------------------------------------------------
    def tinted_glow(self, col, size):
        key = (col, size)
        s = self.scale_cache.get(key)
        if s is None:
            if len(self.scale_cache) > 120:
                self.scale_cache.clear()
            tk = self.tint_cache.get(col)
            if tk is None:
                tk = self.glow_base.copy()
                tk.fill(col + (255,), special_flags=pygame.BLEND_RGBA_MULT)
                self.tint_cache[col] = tk
            s = pygame.transform.smoothscale(tk, (size, size))
            self.scale_cache[key] = s
        return s

    def small_glow(self, r, col):
        key = (r, col)
        s = self.body_glow.get(key)
        if s is None:
            if len(self.body_glow) > 300:
                self.body_glow.clear()
            t = self.glow_small.copy()
            t.fill(col + (255,), special_flags=pygame.BLEND_RGBA_MULT)
            s = pygame.transform.smoothscale(t, (r * 5, r * 5))
            self.body_glow[key] = s
        return s

    def make_bg(self):
        w, h = self.screen.get_size()
        bg = pygame.Surface((w, h))
        for y in range(0, h, 4):
            k = y / h
            pygame.draw.rect(bg, (4 + int(6 * k), 5 + int(6 * k), 12 + int(14 * k)), (0, y, w, 4))
        rng = np.random.default_rng(11)
        for _ in range(int(w * h / 3500)):
            x, y = int(rng.random() * w), int(rng.random() * h)
            b = int(80 + 170 * rng.random() ** 3)
            bg.set_at((x, y), (b, b, min(255, b + 20)))
            if rng.random() < 0.05:
                pygame.draw.circle(bg, (b // 2, b // 2, b // 2), (x, y), 1)
        self.bg, self.bg_size = bg, (w, h)

    # ---- events ---------------------------------------------------------
    def nearest_body(self, spos, maxd=16):
        sim = self.sim
        sp = (sim.pos - self.cam) * self.zoom + self.center()
        d = np.hypot(sp[:, 0] - spos[0], sp[:, 1] - spos[1])
        rad = np.maximum(1.5, 1.35 * sim.m ** (1 / 3) * self.zoom ** 0.85)
        d = d - rad
        i = int(np.argmin(d))
        return i if d[i] < maxd else None

    def predict(self, p0, v0, steps=280, dt=0.6):
        sim = self.sim
        sx, sy = sim.pos[0]
        mu = G * sim.m[0]
        rs = sim.star_radius(sim.m[0])
        px, py = float(p0[0]), float(p0[1])
        vx, vy = float(v0[0]), float(v0[1])
        pts = [(px, py)]
        for _ in range(steps):
            rx, ry = px - sx, py - sy
            r2 = rx * rx + ry * ry + EPS * EPS
            k = -mu / (r2 * math.sqrt(r2))
            vx += 0.5 * dt * k * rx
            vy += 0.5 * dt * k * ry
            px += dt * vx
            py += dt * vy
            rx, ry = px - sx, py - sy
            r2 = rx * rx + ry * ry + EPS * EPS
            k = -mu / (r2 * math.sqrt(r2))
            vx += 0.5 * dt * k * rx
            vy += 0.5 * dt * k * ry
            pts.append((px, py))
            if r2 < rs * rs:
                break
        return pts

    def launch_velocity(self, press, cur):
        d = (np.asarray(cur, float) - np.asarray(press, float)) / self.zoom
        return self.sim.vel[0] + d * LAUNCH_K

    def handle_click(self, spos):
        mods = pygame.key.get_mods()
        i = self.nearest_body(spos)
        if i is not None and not (mods & pygame.KMOD_SHIFT):
            self.selected = int(self.sim.ids[i])
            self.mode = "body"
            return
        w = self.s2w(spos)
        if mods & pygame.KMOD_SHIFT:
            rng = self.sim.rng
            for _ in range(20):
                off = rng.standard_normal(2) * (18.0 / self.zoom)
                p = w + off
                v = self.sim.circular_velocity(p) * (1 + 0.03 * rng.standard_normal())
                self.sim.add_body(p, v, rng.uniform(0.6, 1.6))
            return
        name, m = SPAWN[self.spawn_key]
        if not self.sim.add_circular(w, m):
            self.say("too close to the star (or body limit reached)")

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.VIDEORESIZE:
                self.bg = None
            elif e.type == pygame.KEYDOWN:
                if self.naming:
                    if e.key == pygame.K_RETURN:
                        i = self.sim.index_of(self.selected) if self.selected is not None else None
                        self.sim.set_name(i, self.name_buffer)
                        self.naming = False
                    elif e.key == pygame.K_ESCAPE:
                        self.naming = False
                    elif e.key == pygame.K_BACKSPACE:
                        self.name_buffer = self.name_buffer[:-1]
                    elif e.unicode and e.unicode.isprintable() and len(self.name_buffer) < 24:
                        self.name_buffer += e.unicode
                elif not self.on_key(e):
                    return False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.panning = True
                    else:
                        self.press, self.dragging = e.pos, False
                elif e.button == 2:
                    self.panning = True
                elif e.button == 3:
                    self.attract = True
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    if self.panning:
                        self.panning = False
                    elif self.press is not None:
                        if self.dragging:
                            w = self.s2w(self.press)
                            m = SPAWN[self.spawn_key][1]
                            self.sim.add_body(w, self.launch_velocity(self.press, e.pos), m)
                        else:
                            self.handle_click(e.pos)
                        self.press, self.dragging = None, False
                elif e.button == 2:
                    self.panning = False
                elif e.button == 3:
                    self.attract = False
            elif e.type == pygame.MOUSEMOTION:
                if self.panning:
                    self.cam -= np.array(e.rel, float) / self.zoom
                    self.mode = "free"
                elif self.press is not None and not self.dragging:
                    if math.hypot(e.pos[0] - self.press[0], e.pos[1] - self.press[1]) > 6:
                        self.dragging = True
            elif e.type == pygame.MOUSEWHEEL:
                f = 1.12 ** e.y
                mp = pygame.mouse.get_pos()
                before = self.s2w(mp)
                self.zoom = float(np.clip(self.zoom * f, 0.12, 14.0))
                if self.mode == "free":
                    self.cam = before - (np.array(mp, float) - self.center()) / self.zoom
                self.autozoom = False
        return True

    def on_key(self, e):
        k = e.key
        if k == pygame.K_ESCAPE:
            return False
        if k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_UP:
            self.substeps = min(10, self.substeps + 1)
        elif k == pygame.K_DOWN:
            self.substeps = max(1, self.substeps - 1)
        elif k == pygame.K_LEFTBRACKET:
            self.sim.cap = max(1.0, self.sim.cap - 0.5)
            self.say("accretion radius x%.1f" % self.sim.cap)
        elif k == pygame.K_RIGHTBRACKET:
            self.sim.cap = min(12.0, self.sim.cap + 0.5)
            self.say("accretion radius x%.1f" % self.sim.cap)
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.spawn_key = int(e.unicode)
            self.say("spawn: %s (m=%g)" % SPAWN[self.spawn_key])
        elif k == pygame.K_t:
            self.show_trails = not self.show_trails
        elif k == pygame.K_l:
            self.show_frost = not self.show_frost
        elif k == pygame.K_g:
            self.show_nebula = not self.show_nebula
        elif k == pygame.K_a:
            self.autozoom = not self.autozoom
            self.say("auto-zoom " + ("on" if self.autozoom else "off"))
        elif k == pygame.K_f:
            self.selected = None
            self.mode = "star"
        elif k == pygame.K_d:
            self.sim.add_dust_ring(60)
        elif k == pygame.K_z:
            self.sim.add_gas_cloud(90, 500.0)
            self.say("raw molecular cloud added - build the star")
        elif k == pygame.K_u:
            self.sim.add_star_mass(25.0)
            self.say("25 mass units fed into the protostar")
        elif k == pygame.K_k:
            if self.sim.m[0] >= 8:
                self.sim.collapse_to_black_hole("manual failed-star collapse")
                self.say("stellar collapse: BLACK HOLE")
            else:
                self.say("star needs more mass before collapse")
        elif k == pygame.K_h:
            self.show_help = not self.show_help
        elif k == pygame.K_r:
            self.sim.reset()
            self.selected, self.mode, self.autozoom = None, "star", True
            self.neb_r = 500.0
        elif k == pygame.K_s:
            fn = "solar_system_%d.png" % pygame.time.get_ticks()
            pygame.image.save(self.screen, fn)
            try:
                self.sim.save()
                self.say("system saved to " + SAVE_FILE)
            except Exception as ex:
                self.say("save error: " + str(ex)[:45])
        elif k == pygame.K_F2:
            if self.selected is not None:
                self.naming = True
                self.name_buffer = self.sim.names.get(int(self.selected), "")
        elif k == pygame.K_c:
            self.creator = not self.creator
            self.say("creator mode " + ("ON" if self.creator else "OFF"))
        elif k == pygame.K_DELETE or k == pygame.K_x:
            if self.selected is not None:
                i = self.sim.index_of(self.selected)
                if self.sim.delete_body(i):
                    self.selected = None
                    self.mode = "star"
                    self.say("body deleted")
        elif k in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            i = self.sim.index_of(self.selected) if self.selected is not None else None
            self.sim.change_mass(i, 1.15)
            self.say("mass +15%%")
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            i = self.sim.index_of(self.selected) if self.selected is not None else None
            self.sim.change_mass(i, 1 / 1.15)
            self.say("mass -13%%")
        elif k == pygame.K_o:
            i = self.sim.index_of(self.selected) if self.selected is not None else None
            self.sim.set_circular(i)
            self.say("circular orbit set")
        elif k == pygame.K_v:
            i = self.sim.index_of(self.selected) if self.selected is not None else None
            if i is not None and i > 0:
                self.sim.vel[i] = 2 * self.sim.vel[0] - self.sim.vel[i]
                self.sim.a = None
                self.say("velocity reversed")
        elif k == pygame.K_m:
            i = self.sim.index_of(self.selected) if self.selected is not None else None
            if self.sim.add_moon(i):
                self.say("moon added")
        elif k == pygame.K_b:
            self.sim.add_belt(50)
            self.say("asteroid belt added")
        elif k == pygame.K_F5:
            try:
                self.sim.save()
                self.say("system saved")
            except Exception as ex:
                self.say("save error: " + str(ex)[:45])
        elif k == pygame.K_F9:
            try:
                self.sim.load()
                self.selected, self.mode = None, "star"
                self.say("system loaded")
            except FileNotFoundError:
                self.say("no " + SAVE_FILE + " found")
            except Exception as ex:
                self.say("load error: " + str(ex)[:45])
        elif k == pygame.K_PAGEUP:
            self.time_scale = min(5.0, self.time_scale * 1.25)
        elif k == pygame.K_PAGEDOWN:
            self.time_scale = max(0.1, self.time_scale / 1.25)
        return True

    # ---- update ---------------------------------------------------------
    def update(self):
        sim = self.sim
        if self.attract:
            sign = -1.0 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1.0
            w = self.s2w(pygame.mouse.get_pos())
            sim.mouse = (float(w[0]), float(w[1]), sign)
            sim.a = None
        elif sim.mouse is not None:
            sim.mouse = None
            sim.a = None

        if not self.paused:
            for _ in range(self.substeps):
                sim.step(DT * self.time_scale)
            if self.frame % 2 == 0:
                sim.record_trails()

        # selection follows merges
        if self.selected is not None:
            while self.selected in sim.remap:
                self.selected = sim.remap[self.selected]
            if sim.index_of(self.selected) is None:
                self.selected, self.mode = None, "star"
        sim.remap = {k: v for k, v in sim.remap.items()} if len(sim.remap) < 5000 else {}

        # camera
        if self.mode == "star":
            target = sim.pos[0]
        elif self.mode == "body":
            target = sim.pos[sim.index_of(self.selected)]
        else:
            target = None
        if target is not None:
            self.cam += (target - self.cam) * 0.2
        if self.autozoom:
            rel = sim.pos[1:] - sim.pos[0]
            d = np.hypot(rel[:, 0], rel[:, 1]) if len(rel) else np.array([300.0])
            R = float(np.clip(np.percentile(d, 90), 110, 900))
            w, h = self.screen.get_size()
            tz = float(np.clip(0.44 * min(w, h) / R, 0.2, 6.0))
            self.zoom += (tz - self.zoom) * 0.03
        self.frame += 1

    # ---- drawing --------------------------------------------------------
    def text(self, s, pos, font=None, col=(220, 228, 240), shadow=True):
        font = font or self.font
        if shadow:
            self.screen.blit(font.render(s, True, (0, 0, 0)), (pos[0] + 1, pos[1] + 1))
        self.screen.blit(font.render(s, True, col), pos)

    def panel(self, rect, alpha=150):
        s = pygame.Surface(rect[2:], pygame.SRCALPHA)
        s.fill((8, 12, 24, alpha))
        pygame.draw.rect(s, (70, 90, 130, 180), s.get_rect(), 1)
        self.screen.blit(s, rect[:2])

    def draw_nebula(self):
        sim = self.sim
        vis = sim.gas_visual()
        if vis < 0.01:
            return
        rel = sim.pos[1:] - sim.pos[0]
        d = np.hypot(rel[:, 0], rel[:, 1]) if len(rel) else np.array([200.0])
        Rn = 1.4 * float(np.percentile(d, 92))
        self.neb_r += (Rn - self.neb_r) * 0.04
        size = int(2 * self.neb_r * self.zoom)
        size = max(32, min(size, 1800)) // 16 * 16
        surf = self.neb_cache.get(size)
        if surf is None:
            if len(self.neb_cache) > 40:
                self.neb_cache.clear()
            surf = pygame.transform.smoothscale(self.nebula, (size, size))
            self.neb_cache[size] = surf
        surf.set_alpha(int(215 * vis))
        c = self.w2s(sim.pos[0])
        self.screen.blit(surf, (int(c[0] - size / 2), int(c[1] - size / 2)))

    def draw_star(self):
        sim, scr = self.sim, self.screen
        M = sim.m[0]
        c = self.w2s(sim.pos[0])
        if sim.black_hole:
            # Stylized gravitational lensing / event-horizon view. The actual
            # particle dynamics remain Newtonian; this makes the spacetime
            # curvature visible without pretending to solve full GR.
            rs = max(2.0, sim.schwarzschild_radius(M) * self.zoom)
            for q, alpha in ((5.5, 45), (3.5, 70), (2.0, 110)):
                rr = int(max(8, rs * q))
                ring = pygame.Surface((rr * 2 + 8, rr * 2 + 8), pygame.SRCALPHA)
                pygame.draw.circle(ring, (100, 150, 255, alpha), (rr + 4, rr + 4), rr, 2)
                scr.blit(ring, (int(c[0] - rr - 4), int(c[1] - rr - 4)))
            pygame.draw.circle(scr, (2, 2, 5), (int(c[0]), int(c[1])), int(max(5, rs * 1.8)))
            pygame.draw.circle(scr, (240, 210, 120), (int(c[0]), int(c[1])), int(max(6, rs * 2.7)), 2)
            self.text("EVENT HORIZON", (int(c[0]) + 12, int(c[1]) - 12), self.small, (255, 210, 130))
            return
        f = min(1.0, max(0.0, (M - 1.0) / max(1.0, STAR_M1 - 1.0)))
        q = round(f * 10) / 10
        col = (255, int(80 + 155 * q), int(30 + 170 * q))
        rpx = max(2.0, Sim.star_radius(M) * self.zoom)
        size = int(max(48, rpx * 9)) // 8 * 8
        size = min(size, 1600)
        flick = 0.93 + 0.07 * math.sin(sim.t * 0.9) * math.sin(sim.t * 0.37)
        g = self.tinted_glow(col, size)
        g.set_alpha(int(255 * flick))
        scr.blit(g, (int(c[0] - size / 2), int(c[1] - size / 2)))
        pygame.draw.circle(scr, col, (int(c[0]), int(c[1])), int(rpx))
        pygame.draw.circle(scr, (255, 252, 240), (int(c[0]), int(c[1])), max(1, int(rpx * 0.72)))
    def draw_trails(self, cls, icy):
        sim, scr = self.sim, self.screen
        n = len(sim.m)
        hs = (sim.hist + sim.pos[0] - self.cam) * self.zoom + self.center()
        hs = np.clip(hs, -20000, 20000).astype(int)
        L = hs.shape[1]
        cuts = ((0, L // 3 + 1, 0.22), (L // 3, 2 * L // 3 + 1, 0.45), (2 * L // 3, L, 0.8))
        w, h = scr.get_size()
        for i in range(1, n):
            if cls[i] == 0 and n > 260:
                continue
            p = hs[i]
            if p[:, 0].max() < 0 or p[:, 0].min() > w or p[:, 1].max() < 0 or p[:, 1].min() > h:
                continue
            base = COL[cls[i]][int(icy[i])]
            for a, b, s in cuts:
                pygame.draw.lines(scr, (int(base[0] * s), int(base[1] * s), int(base[2] * s)),
                                  False, p[a:b].tolist(), 1)

    def draw_bodies(self, cls, icy, hover):
        sim, scr = self.sim, self.screen
        w, h = scr.get_size()
        sp = (sim.pos - self.cam) * self.zoom + self.center()
        rad = np.maximum(1.5, 1.35 * sim.m ** (1 / 3) * self.zoom ** 0.85)
        for i in range(1, len(sim.m)):
            x, y = sp[i]
            if x < -40 or y < -40 or x > w + 40 or y > h + 40:
                continue
            col = COL[cls[i]][int(icy[i])]
            r = int(rad[i])
            if cls[i] >= 2:
                gl = self.small_glow(max(2, r), col)
                scr.blit(gl, (int(x - gl.get_width() / 2), int(y - gl.get_height() / 2)))
            pygame.draw.circle(scr, col, (int(x), int(y)), max(1, r))
            if cls[i] >= 3:
                pygame.draw.circle(scr, (255, 255, 255), (int(x - r * 0.3), int(y - r * 0.3)),
                                   max(1, r // 4))
            if sim.ids[i] == self.selected:
                pygame.draw.circle(scr, (255, 255, 255), (int(x), int(y)), r + 6, 1)
                pygame.draw.circle(scr, (120, 200, 255), (int(x), int(y)), r + 10, 1)
            elif i == hover:
                pygame.draw.circle(scr, (200, 220, 255), (int(x), int(y)), r + 5, 1)

    def draw_overlays(self):
        scr, sim = self.screen, self.sim
        mp = pygame.mouse.get_pos()
        if self.attract:
            sign = -1 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
            ph = (self.frame % 40) / 40.0
            col = (255, 120, 90) if sign < 0 else (120, 200, 255)
            for k in range(3):
                t = (ph + k / 3.0) % 1.0
                r = int(8 + 60 * (t if sign < 0 else 1 - t))
                pygame.draw.circle(scr, col, mp, r, 1)
            pygame.draw.circle(scr, col, mp, 4)
        if self.press is not None and self.dragging:
            w0 = self.s2w(self.press)
            v0 = self.launch_velocity(self.press, mp)
            pts = self.predict(w0, v0)
            sp = np.clip(np.array([self.w2s(p) for p in pts]), -20000, 20000).astype(int)
            for k in range(0, len(sp) - 1, 2):
                pygame.draw.line(scr, (255, 230, 120), sp[k], sp[k + 1], 1)
            pygame.draw.line(scr, (255, 255, 255), self.press, mp, 1)
            pygame.draw.circle(scr, (255, 255, 255), self.press, 5, 1)
            vv = np.hypot(*(v0 - sim.vel[0]))
            self.text("v = %.1f" % vv, (mp[0] + 12, mp[1] + 8), self.small)

    def draw_hud(self, hover):
        scr, sim = self.screen, self.sim
        w, h = scr.get_size()
        k, name = stage_of(sim.t)
        self.panel((10, 10, 340, 128))
        self.text("SOLAR SYSTEM GENESIS", (20, 16), self.big, (255, 220, 150))
        state = sim.stellar_state()
        self.text("Stage " + name + " | " + state, (20, 44), self.font, (150, 220, 255))
        # progress bar
        bx, by, bw = 20, 68, 320
        pygame.draw.rect(scr, (30, 40, 60), (bx, by, bw, 8))
        frac = min(1.0, sim.t / T_CLEAR)
        pygame.draw.rect(scr, (120, 190, 255), (bx, by, int(bw * frac), 8))
        for edge in (T_COLLAPSE, T_DISPERSE):
            xx = bx + int(bw * edge / T_CLEAR)
            pygame.draw.line(scr, (255, 255, 255), (xx, by - 2), (xx, by + 10), 1)
        n = len(sim.m) - 1
        planets = int((sim.m[1:] >= 60).sum())
        self.text("t = %6.1f Myr   speed x%d%s" % (sim.t * T_TO_MYR, self.substeps,
                                                  "  [PAUSED]" if self.paused else ""),
                  (20, 84), self.small)
        self.text("bodies %d  planets %d  star mass %.1f  fuel %.0f%%" % (n, planets, sim.m[0], 100*sim.star_fuel),
                  (20, 102), self.small)
        self.text("fps %d   spawn: %s   accretion x%.1f   sim x%.1f" % (
            self.clock.get_fps(), SPAWN[self.spawn_key][0], sim.cap, self.time_scale), (20, 118), self.small,
            (150, 160, 180))

        if self.notice[1] > 0:
            s, t = self.notice
            self.text(s, (w // 2 - 6 * len(s) // 1 // 2 * 1, h - 44), self.font, (255, 230, 150))
            self.notice = (s, t - 1)

        idx = None
        if self.selected is not None:
            idx = sim.index_of(self.selected)
        elif hover is not None:
            idx = hover
        if idx is not None:
            self.draw_info(idx)

        if self.creator:
            cx, cy, cw, ch = 370, 10, 315, 170
            self.panel((cx, cy, cw, ch), 145)
            self.text("CREATOR MODE", (cx + 10, cy + 8), self.font, (255, 220, 150))
            selected_name = "none"
            if self.selected is not None:
                ii = sim.index_of(self.selected)
                if ii is not None:
                    selected_name = sim.body_name(ii)
            creator_lines = [
                "Selected: " + selected_name,
                "1-4 spawn type | click = orbit",
                "Z raw gas cloud | U feed star",
                "K force failed-star collapse",
                "drag = custom launch velocity",
                "+/- mass | O circular orbit | V reverse",
                "M moon | B asteroid belt | X delete",
                "F2 rename selected body",
                "F5 save | F9 load",
                "PgUp/PgDn simulation speed",
                "Right mouse = gravity / Shift = repel",
            ]
            for j, ln in enumerate(creator_lines):
                self.text(ln, (cx + 10, cy + 34 + j * 15), self.small, (200, 210, 225))

        if self.naming:
            self.panel((w // 2 - 190, h // 2 - 45, 380, 90), 220)
            self.text("Rename selected body", (w // 2 - 170, h // 2 - 32), self.font, (255, 220, 150))
            self.text(self.name_buffer + "_", (w // 2 - 170, h // 2), self.font, (240, 245, 255))
            self.text("Enter = save   Esc = cancel", (w // 2 - 170, h // 2 + 25), self.small, (170, 180, 200))

        if self.show_help:
            lines = [
                "MOUSE",
                " click empty     spawn body (orbiting)",
                " drag            slingshot launch",
                " shift+click     dust burst",
                " click body      select / follow",
                " right hold      attract  (+shift: repel)",
                " wheel           zoom",
                " middle/ctrl+drag  pan",
                "KEYS",
                " SPACE pause   UP/DOWN speed",
                " [ ]  accretion radius",
                " 1-4  spawn type   D dust",
                " T trails  L frost line  G gas",
                " A autozoom  F follow star",
                " R restart  S screenshot",
                " H hide help   ESC quit",
            ]
            ph = 16 * len(lines) + 14
            self.panel((10, h - ph - 10, 300, ph), 130)
            for i, ln in enumerate(lines):
                col = (255, 220, 150) if ln in ("MOUSE", "KEYS") else (200, 210, 225)
                self.text(ln, (18, h - ph - 3 + i * 16), self.small, col)
        else:
            self.text("H: help", (14, h - 24), self.small, (150, 160, 180))

    def draw_info(self, i):
        sim, scr = self.sim, self.screen
        w, h = scr.get_size()
        lines = []
        if i == 0:
            lines.append((sim.body_name(i), (255, 220, 150)))
            lines.append(("state   " + sim.stellar_state(), None))
            lines.append(("mass    %.1f" % sim.m[0], None))
            lines.append(("fuel    %.0f%%" % (100*sim.star_fuel), None))
            lines.append(("swallowed %.1f" % sim.absorbed, None))
            if sim.black_hole:
                lines.append(("Rs      %.2f" % sim.schwarzschild_radius(sim.m[0]), (255, 190, 120)))
        else:
            info = sim.orbit_info(i)
            icy = info["r"] > FROST
            c, tname = classify(sim.m[i], icy)
            lines.append((sim.body_name(i), COL[c][int(icy)]))
            lines.append((tname + ("  (icy)" if icy and c < 3 else ""), COL[c][int(icy)]))
            lines.append(("mass    %.1f" % sim.m[i], None))
            lines.append(("dist    %.0f" % info["r"], None))
            lines.append(("speed   %.2f" % info["v"], None))
            if "a" in info:
                lines.append(("semi-axis %.0f" % info["a"], None))
                lines.append(("ecc.    %.2f" % info["e"], None))
                lines.append(("period  %.0f" % info["T"], None))
            else:
                lines.append(("unbound orbit", (255, 150, 120)))
        pw, ph = 190, 14 + 17 * len(lines)
        x0, y0 = w - pw - 10, 10
        self.panel((x0, y0, pw, ph))
        for k, (s, col) in enumerate(lines):
            self.text(s, (x0 + 10, y0 + 7 + 17 * k), self.small, col or (215, 225, 240))

    def draw(self):
        scr, sim = self.screen, self.sim
        if self.bg is None or self.bg_size != scr.get_size():
            self.make_bg()
        scr.blit(self.bg, (0, 0))

        if self.show_nebula:
            self.draw_nebula()

        if self.show_frost:
            c = self.w2s(sim.pos[0])
            r = int(FROST * self.zoom)
            if 4 < r < 20000:
                pygame.draw.circle(scr, (80, 130, 190), (int(c[0]), int(c[1])), r, 1)
                self.text("frost line", (int(c[0]) + 6, int(c[1]) - r - 16), self.small,
                          (110, 160, 220))

        rel = sim.pos - sim.pos[0]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        icy = dist > FROST
        for ii, ident in enumerate(sim.ids):
            if int(ident) in sim.custom_icy:
                icy[ii] = sim.custom_icy[int(ident)]
        cls = np.digitize(sim.m, (2.5, 15, 60, 150))

        hover = None
        if not (self.press is not None and self.dragging):
            hover = self.nearest_body(pygame.mouse.get_pos())

        if self.show_trails:
            self.draw_trails(cls, icy)
        self.draw_bodies(cls, icy, hover)
        self.draw_star()
        self.draw_overlays()
        self.draw_hud(hover)
        pygame.display.flip()

    # ---- main loop ------------------------------------------------------
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
