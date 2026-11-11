"""
[NEW FILE] One experiment = simulation settings + controller config + foot
placement policy + velocity command. Plain, picklable data, so run.py,
poincare_aug.py and roa.py (and their worker processes) build the exact same
setup.

Command line (shared by all three programs):
    --vx --vy --wz --policy --k
    --config cfg.json                       JSON with sections (see below)
    --set mpc.h=30 sim.friction=0.8 ...      any dataclass field, repeatable
    sim shortcuts: --dt --stiffness --damping --solref --solimp --friction
                   --condim --cone --integrator --impratio --obstacles
Sections for --set / JSON: sim (SimCfg), mpc (MPC_Cfg), gait (GaitCfg),
swing (SwingTrajectoryCfg), leg (LegControllerCfg), ctrl (ControllerCfg),
exp (this class: policy, policy_kw, v_cmd, t_ramp, periodic_ref_path, mu_auto).
"""
import argparse
import ast
import copy
import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from structs import SimCfg, ControllerCfg
from foot_place import POLICIES


@dataclass
class Experiment:
    sim: SimCfg = field(default_factory=SimCfg)
    ctrl: ControllerCfg = field(default_factory=ControllerCfg)
    policy: str = "raibert"
    policy_kw: dict = field(default_factory=lambda: {"k": 0.03})
    v_cmd: tuple = (0.5, 0.0, 0.0)          # [vx, vy, wz] heading frame
    t_ramp: float = 1.0                     # command ramp after standing [s]
    periodic_ref_path: str | None = None    # PeriodicReference .npz (enables mpc.periodic_ref)
    # MPC friction pyramid from the simulated contact: mu = friction / sqrt(2)
    # (pyramid inscribed in MuJoCo's round cone). False: keep mpc.mu.
    mu_auto: bool = True

    @property
    def label(self) -> str:
        kw = ",".join(f"{k}={v}" for k, v in self.policy_kw.items())
        return f"{self.policy}({kw})"

    def build(self):
        """-> (model, data, controller)"""
        from sim_env import make_model, foot_friction
        from quad_ctrl import QuadrupedController
        from ref_gen import PeriodicReference
        model, data = make_model(self.sim)
        cfg = copy.deepcopy(self.ctrl)
        if self.mu_auto:
            cfg.mpc_cfg.mu = foot_friction(model) / np.sqrt(2.0)
        pref = None
        if self.periodic_ref_path:
            pref = PeriodicReference.load(self.periodic_ref_path)
            cfg.mpc_cfg.periodic_ref = True
        ctrl = QuadrupedController(model, data, cfg, POLICIES[self.policy](**self.policy_kw), pref)
        return model, data, ctrl

    def command(self):
        v = np.asarray(self.v_cmd, dtype=float)
        t0, tr = self.ctrl.t_stand, self.t_ramp
        return lambda t: v * np.clip((t - t0) / tr, 0.0, 1.0) if tr > 0 else v * (t >= t0)

    # ---- (de)serialization ----
    def to_dict(self) -> dict:
        """Same section layout that apply_dict / --config read."""
        c = self.ctrl
        cfg_names = {"mpc_cfg", "gait_cfg", "swing_cfg", "leg_cfg"}
        d = {"sim": dataclasses.asdict(self.sim),
             "mpc": dataclasses.asdict(c.mpc_cfg), "gait": dataclasses.asdict(c.gait_cfg),
             "swing": dataclasses.asdict(c.swing_cfg), "leg": dataclasses.asdict(c.leg_cfg),
             "ctrl": {f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name not in cfg_names},
             "exp": {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                     if f.name not in ("sim", "ctrl")}}
        return _to_jsonable(d)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------
def _section(exp: Experiment, name: str):
    return {"sim": exp.sim, "ctrl": exp.ctrl, "mpc": exp.ctrl.mpc_cfg,
            "gait": exp.ctrl.gait_cfg, "swing": exp.ctrl.swing_cfg,
            "leg": exp.ctrl.leg_cfg, "exp": exp}[name]


def _parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text            # plain string, e.g. hard / elliptic


def set_field(exp: Experiment, key: str, value):
    """key = 'section.field', e.g. 'mpc.h'. Lists become numpy arrays where the default is one."""
    sec, _, name = key.partition(".")
    obj = _section(exp, sec)
    if not hasattr(obj, name):
        fields = [f.name for f in dataclasses.fields(obj)]
        raise KeyError(f"{sec} has no field '{name}'. Fields: {fields}")
    old = getattr(obj, name)
    if isinstance(old, np.ndarray):
        value = np.asarray(value, dtype=float)
    elif isinstance(value, list) and name not in ("policy_kw",):
        value = tuple(value)
    setattr(obj, name, value)
    if sec == "gait":                      # GaitCfg normalizes the table
        obj.__post_init__()


def apply_dict(exp: Experiment, d: dict):
    """JSON layout: {"sim": {...}, "mpc": {...}, "exp": {...}, ...}"""
    for sec, fields in d.items():
        for k, v in fields.items():
            set_field(exp, f"{sec}.{k}", v)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def add_cli(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("experiment")
    g.add_argument("--vx", type=float, default=None)
    g.add_argument("--vy", type=float, default=None)
    g.add_argument("--wz", type=float, default=None)
    g.add_argument("--policy", choices=sorted(POLICIES), default=None)
    g.add_argument("--k", type=float, default=None, help="foot placement feedback gain")
    g.add_argument("--config", default=None, help="JSON file with sections sim/mpc/gait/swing/leg/ctrl/exp")
    g.add_argument("--set", nargs="+", default=[], metavar="SEC.FIELD=VALUE",
                   help="override any config field, e.g. mpc.h=30 mpc.period_constraint=hard")
    g.add_argument("--periodic-ref", default=None, help="PeriodicReference .npz -> periodic MPC reference")
    s = ap.add_argument_group("simulation (MuJoCo) settings")
    s.add_argument("--dt", type=float, default=None, help="sim timestep [s]")
    s.add_argument("--stiffness", type=float, default=None, help="contact stiffness k (direct solref=(-k,-b))")
    s.add_argument("--damping", type=float, default=None, help="contact damping b (with --stiffness)")
    s.add_argument("--solref", type=float, nargs=2, default=None, metavar=("TIMECONST", "DAMPRATIO"))
    s.add_argument("--solimp", type=float, nargs=5, default=None, metavar=("DMIN", "DMAX", "WIDTH", "MID", "POW"))
    s.add_argument("--friction", type=float, default=None, help="floor+foot sliding friction")
    s.add_argument("--condim", type=int, default=None, choices=[1, 3, 4, 6])
    s.add_argument("--cone", choices=["pyramidal", "elliptic"], default=None)
    s.add_argument("--integrator", choices=["euler", "implicitfast", "implicit", "rk4"], default=None)
    s.add_argument("--impratio", type=float, default=None)
    s.add_argument("--obstacles", action="store_true", help="keep bars/stairs of the stock scene")


def from_cli(a: argparse.Namespace) -> Experiment:
    exp = Experiment()
    if a.config:
        with open(a.config) as f:
            apply_dict(exp, json.load(f))
    v = list(exp.v_cmd)
    for i, name in enumerate(("vx", "vy", "wz")):
        if getattr(a, name) is not None:
            v[i] = getattr(a, name)
    exp.v_cmd = tuple(v)
    if a.policy:
        exp.policy = a.policy
        if a.policy == "openloop":
            exp.policy_kw = {}
    if a.k is not None:
        if exp.policy == "openloop":
            raise SystemExit("--k has no effect on the openloop policy")
        exp.policy_kw = {"k": a.k}
    if a.periodic_ref:
        exp.periodic_ref_path = a.periodic_ref
    shortcuts = {"dt": "timestep", "stiffness": "contact_stiffness", "damping": "contact_damping",
                 "solref": "contact_solref", "solimp": "contact_solimp", "friction": "friction",
                 "condim": "condim", "cone": "cone", "integrator": "integrator", "impratio": "impratio"}
    for arg, fld in shortcuts.items():
        val = getattr(a, arg)
        if val is not None:
            set_field(exp, f"sim.{fld}", tuple(val) if isinstance(val, list) else val)
    if a.obstacles:
        exp.sim.obstacles = True
    for item in a.set:
        key, _, val = item.partition("=")
        set_field(exp, key.strip(), _parse_value(val.strip()))
    return exp
