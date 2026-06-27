"""
Mechanics of materials calculation engine.
Computes reactions, SFD, BMD, deflection, and stress for beam structures.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from enum import Enum
from dataclasses import dataclass, field


class BeamType(Enum):
    CANTILEVER = "cantilever"
    SIMPLY_SUPPORTED = "simply_supported"
    FIXED_FIXED = "fixed_fixed"
    OVERHANGING = "overhanging"


class CrossSection(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    I_BEAM = "i_beam"
    HOLLOW_RECT = "hollow_rect"


@dataclass
class BeamModel:
    """Simplified beam model extracted from detections."""

    length: float  # m
    beam_type: BeamType
    supports: list[dict]  # [{"position": 0.0, "type": "fixed"}, ...]
    point_loads: list[dict]  # [{"position": 1.5, "value": -10000}, ...]  N, negative = downward
    distributed_loads: list[dict]  # [{"start": 0, "end": 2.0, "value": -5000}, ...]  N/m
    cross_section: dict = field(default_factory=lambda: {
        "type": CrossSection.RECTANGLE,
        "width": 0.1,   # m
        "height": 0.2,  # m
    })
    material: dict = field(default_factory=lambda: {
        "E": 200e9,     # Young's modulus (Pa), default steel
        "nu": 0.3,      # Poisson's ratio
        "yield_stress": 250e6,  # Pa
    })


@dataclass
class BeamResults:
    """Results of beam analysis."""

    beam: BeamModel
    positions: np.ndarray  # sample positions along beam (m)
    reactions: dict  # reaction forces at each support
    shear: np.ndarray  # shear force at each position (N)
    moment: np.ndarray  # bending moment at each position (N·m)
    deflection: np.ndarray  # deflection at each position (m)
    slope: np.ndarray  # slope at each position (rad)
    bending_stress: np.ndarray  # max bending stress at each position (Pa)
    max_deflection: float
    max_moment: float
    max_shear: float
    max_stress: float


class BeamAnalyzer:
    """
    Analyzes beam structures using mechanics of materials.
    Supports cantilever, simply-supported, fixed-fixed, and overhanging beams.
    """

    N_POINTS = 500  # resolution for diagrams

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, beam: BeamModel) -> BeamResults:
        """Run full analysis on a beam model."""
        x = np.linspace(0, beam.length, self.N_POINTS)

        # 1. classify beam type from supports
        beam.beam_type = self._classify_beam(beam)

        # 2. compute reaction forces
        reactions = self._solve_reactions(beam)

        # 3. shear force diagram
        V = self._compute_shear(beam, reactions, x)

        # 4. bending moment diagram  (numerical integration of shear)
        M = self._compute_moment(V, x)
        M = self._apply_moment_bc(beam, M, x)

        # 5. deflection via double-integration
        dv, v = self._compute_deflection(beam, M, x)

        # 6. bending stress  σ = M·y / I
        I = self._moment_of_inertia(beam.cross_section)
        y_max = beam.cross_section.get("height", 0.2) / 2
        sigma = np.abs(M) * y_max / I

        return BeamResults(
            beam=beam,
            positions=x,
            reactions=reactions,
            shear=V,
            moment=M,
            deflection=v,
            slope=dv,
            bending_stress=sigma,
            max_deflection=float(np.max(np.abs(v))),
            max_moment=float(np.max(np.abs(M))),
            max_shear=float(np.max(np.abs(V))),
            max_stress=float(np.max(sigma)),
        )

    # ------------------------------------------------------------------
    # Beam classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_beam(beam: BeamModel) -> BeamType:
        supports = beam.supports
        n = len(supports)
        if n == 0:
            raise ValueError("Beam must have at least one support")
        if n == 1:
            return BeamType.CANTILEVER
        if n == 2:
            types = [s["type"] for s in supports]
            if types.count("fixed") == 2:
                return BeamType.FIXED_FIXED
            return BeamType.SIMPLY_SUPPORTED
        if n > 2:
            return BeamType.OVERHANGING
        return BeamType.SIMPLY_SUPPORTED

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    def _solve_reactions(self, beam: BeamModel) -> dict:
        """Solve for reaction forces using static equilibrium."""
        L = beam.length
        supports = sorted(beam.supports, key=lambda s: s["position"])

        # sum of vertical forces + reactions = 0
        total_load = self._total_vertical_load(beam)

        # moment about first support
        total_moment = self._total_moment_about(beam, supports[0]["position"])

        if beam.beam_type == BeamType.CANTILEVER:
            return self._solve_cantilever(beam, total_load, total_moment)

        if beam.beam_type in (BeamType.SIMPLY_SUPPORTED, BeamType.FIXED_FIXED):
            return self._solve_two_support(beam, supports, total_load, total_moment)

        # overhanging — statically determinate with 3+ supports needs assumptions
        # treat as simply-supported with two outermost supports
        outer = [supports[0], supports[-1]]
        total_moment = self._total_moment_about(beam, outer[0]["position"])
        return self._solve_two_support(beam, outer, total_load, total_moment)

    def _solve_cantilever(self, beam, total_load, total_moment):
        """Fixed end: reaction = -total_load, moment reaction = -total_moment."""
        s = beam.supports[0]
        return {
            (s["position"], s["type"]): {
                "Ry": -total_load,
                "M_reaction": -total_moment,
            }
        }

    def _solve_two_support(self, beam, supports, total_load, total_moment):
        L = supports[1]["position"] - supports[0]["position"]
        # moment about left support: R_right * L + total_moment = 0
        R_right = -total_moment / L
        R_left = -total_load - R_right

        return {
            (supports[0]["position"], supports[0]["type"]): {"Ry": R_left},
            (supports[1]["position"], supports[1]["type"]): {"Ry": R_right},
        }

    def _total_vertical_load(self, beam: BeamModel) -> float:
        total = 0.0
        for pl in beam.point_loads:
            total += pl["value"]
        for dl in beam.distributed_loads:
            length = abs(dl["end"] - dl["start"])
            total += dl["value"] * length
        return total

    def _total_moment_about(self, beam: BeamModel, ref_x: float) -> float:
        total = 0.0
        for pl in beam.point_loads:
            total += pl["value"] * (pl["position"] - ref_x)
        for dl in beam.distributed_loads:
            mid = (dl["start"] + dl["end"]) / 2
            length = abs(dl["end"] - dl["start"])
            total += dl["value"] * length * (mid - ref_x)
        return total

    # ------------------------------------------------------------------
    # Shear force diagram
    # ------------------------------------------------------------------

    def _compute_shear(self, beam: BeamModel, reactions: dict, x: np.ndarray) -> np.ndarray:
        V = np.zeros_like(x)

        # reaction contributions
        for (pos, _), r in reactions.items():
            V += np.where(x >= pos, r["Ry"], 0.0)

        # point load contributions
        # pl["value"] is already negative for downward loads — use it directly
        for pl in beam.point_loads:
            V += np.where(x >= pl["position"], pl["value"], 0.0)

        # distributed load contributions — linear ramp
        for dl in beam.distributed_loads:
            a, b = sorted([dl["start"], dl["end"]])
            w = dl["value"]
            for i, xi in enumerate(x):
                if xi < a:
                    continue
                elif xi <= b:
                    V[i] += w * (xi - a)
                else:
                    V[i] += w * (b - a)

        return V

    # ------------------------------------------------------------------
    # Bending moment diagram
    # ------------------------------------------------------------------

    def _compute_moment(self, V: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Integrate shear to get moment: M(x) = ∫₀ˣ V(ξ) dξ."""
        M = cumulative_trapezoid(V, x, initial=0)
        return M

    @staticmethod
    def _apply_moment_bc(beam: BeamModel, M: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Correct moment diagram for beam boundary conditions.

        For simply-supported / fixed-fixed: M(0)=0, M(L)=0
          → enforce zero at both ends via linear correction.

        For cantilever (fixed at left, free at right): M(L)=0
          → flip the moment so M(0) holds the reaction moment
          and M(L) = 0.  (M_raw already integrates to M_raw(L)
          at the free end, so M = M_raw[-1] - M_raw.)
        """
        if beam.beam_type == BeamType.CANTILEVER:
            return M[-1] - M  # M(0) = reaction moment, M(L) = 0
        else:
            # linear correction to zero both ends (handles small numerical drift)
            L = x[-1]
            return M - M[0] - (M[-1] - M[0]) * x / L

    # ------------------------------------------------------------------
    # Deflection & slope  (double integration  EI·v'' = M)
    # ------------------------------------------------------------------

    def _compute_deflection(self, beam: BeamModel, M: np.ndarray, x: np.ndarray):
        """Double-integrate M/EI to get slope and deflection."""
        E = beam.material["E"]
        I = self._moment_of_inertia(beam.cross_section)
        EI = E * I

        slope = cumulative_trapezoid(M, x, initial=0) / EI
        deflection = cumulative_trapezoid(slope, x, initial=0)

        # Apply boundary conditions — zero deflection at supports
        self._apply_bc(beam, x, slope, deflection)

        return slope, deflection

    def _apply_bc(self, beam: BeamModel, x, slope, deflection):
        """
        Enforce deflection and slope boundary conditions.

        Cantilever (1 fixed support): deflection=0, slope=0 at fixed end.
        Simply-supported (2 pinned/roller): deflection=0 at both supports.
          → linear correction: subtract the line through (x0, v0) and (x1, v1).
        Fixed-fixed / overhanging: zero deflection at each support via
          linear correction between outermost supports.
        """
        n = len(beam.supports)
        if n == 0:
            return

        if n == 1 and beam.supports[0]["type"] == "fixed":
            # cantilever: zero deflection AND slope at the fixed end
            idx = np.argmin(np.abs(x - beam.supports[0]["position"]))
            deflection -= deflection[idx]
            slope -= slope[idx]
            return

        # 2+ supports: zero deflection at each support position
        # Use outermost supports for the linear baseline
        sorted_supports = sorted(beam.supports, key=lambda s: s["position"])
        x0 = sorted_supports[0]["position"]
        x1 = sorted_supports[-1]["position"]
        idx0 = np.argmin(np.abs(x - x0))
        idx1 = np.argmin(np.abs(x - x1))

        v0 = deflection[idx0]
        v1 = deflection[idx1]
        L_span = x1 - x0 if x1 > x0 else 1.0

        # linear correction: subtract the line from (x0, v0) to (x1, v1)
        baseline = v0 + (v1 - v0) * (x - x0) / L_span
        deflection -= baseline

        # for fixed supports: also zero slope at those supports
        for s in sorted_supports:
            if s["type"] == "fixed":
                idx = np.argmin(np.abs(x - s["position"]))
                slope -= slope[idx]

    # ------------------------------------------------------------------
    # Cross-section properties
    # ------------------------------------------------------------------

    @staticmethod
    def _moment_of_inertia(cs: dict) -> float:
        """Compute area moment of inertia I for common sections."""
        t = cs["type"]
        if t == CrossSection.RECTANGLE:
            b, h = cs["width"], cs["height"]
            return b * h**3 / 12
        if t == CrossSection.CIRCLE:
            d = cs.get("diameter", 0.1)
            return np.pi * d**4 / 64
        if t == CrossSection.I_BEAM:
            # simplified: flange + web
            bf, tf = cs.get("flange_width", 0.1), cs.get("flange_thickness", 0.01)
            tw, h = cs.get("web_thickness", 0.006), cs.get("height", 0.2)
            hw = h - 2 * tf
            I_flange = 2 * (bf * tf**3 / 12 + bf * tf * ((hw + tf) / 2) ** 2)
            I_web = tw * hw**3 / 12
            return I_flange + I_web
        if t == CrossSection.HOLLOW_RECT:
            bo, ho = cs["outer_width"], cs["outer_height"]
            bi, hi = cs.get("inner_width", bo - 0.02), cs.get("inner_height", ho - 0.02)
            return (bo * ho**3 - bi * hi**3) / 12
        raise ValueError(f"Unknown cross-section type: {t}")
