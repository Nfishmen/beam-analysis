"""
Verify mechanics calculations against analytical solutions.
Run: python verify_mechanics.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.mechanics import BeamModel, BeamAnalyzer, BeamType, CrossSection

analyzer = BeamAnalyzer()
errors = []

# ===========================================================================
# Case 1: Simply-supported beam, point load at midspan
# ===========================================================================
beam1 = BeamModel(
    length=4.0, beam_type=BeamType.SIMPLY_SUPPORTED,
    supports=[{"type": "pinned", "position": 0.0}, {"type": "roller", "position": 4.0}],
    point_loads=[{"position": 2.0, "value": -20000}],
    distributed_loads=[],
    cross_section={"type": CrossSection.RECTANGLE, "width": 0.15, "height": 0.3},
    material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
)
r1 = analyzer.analyze(beam1)

# Analytical solutions
I1 = 0.15 * 0.3**3 / 12
v1_analytical = 20000 * 4.0**3 / (48 * 200e9 * I1)  # PL^3/(48EI)

print("=" * 60)
print("Case 1: Simply-Supported Beam, P=-20kN at midspan")
print("=" * 60)

# Check reactions
print(f"\nReactions: {r1.reactions}")
assert abs(r1.max_shear - 10000) < 1, f"V_max should be 10000, got {r1.max_shear}"

# Check shear values at specific points
idx_left = np.argmin(np.abs(r1.positions - 1.0))
idx_right = np.argmin(np.abs(r1.positions - 3.0))
V_left = r1.shear[idx_left]
V_right = r1.shear[idx_right]
print(f"V@1m: {V_left:.1f} (should be +10000)")
print(f"V@3m: {V_right:.1f} (should be -10000)")

if V_left < 0:
    errors.append("CASE1: V sign flipped in left half — point load sign bug")
if V_right > 0:
    errors.append("CASE1: V sign flipped in right half — point load sign bug")

# Check moment
print(f"M_max: {r1.max_moment:.1f} (should be 20000 N·m)")
print(f"M@x=0: {r1.moment[0]:.1f} (should be 0)")
print(f"M@x=L: {r1.moment[-1]:.1f} (should be 0)")

if abs(r1.max_moment - 20000) / 20000 > 0.05:
    errors.append(f"CASE1: M_max off by >5%: {r1.max_moment:.1f} vs 20000")

if abs(r1.moment[0]) > 1:
    errors.append(f"CASE1: M@0 should be 0, got {r1.moment[0]:.1f}")

# Check deflection
print(f"v_max: {r1.max_deflection*1e3:.4f} mm (should be {v1_analytical*1e3:.4f} mm)")
print(f"v@x=0: {r1.deflection[0]*1e6:.2f} μm (should be 0)")
print(f"v@x=L: {r1.deflection[-1]*1e6:.2f} μm (should be 0)")

if abs(r1.deflection[0]) > 1e-9:
    errors.append(f"CASE1: v@0 should be 0, got {r1.deflection[0]:.2e}")

# ===========================================================================
# Case 2: Cantilever beam, uniform distributed load
# ===========================================================================
beam2 = BeamModel(
    length=2.5, beam_type=BeamType.CANTILEVER,
    supports=[{"type": "fixed", "position": 0.0}],
    point_loads=[],
    distributed_loads=[{"start": 0.0, "end": 2.5, "value": -8000}],
    cross_section={"type": CrossSection.RECTANGLE, "width": 0.1, "height": 0.2},
    material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
)
r2 = analyzer.analyze(beam2)

I2 = 0.1 * 0.2**3 / 12
M2_fixed = 8000 * 2.5**2 / 2  # wL^2/2 = 25000
v2_tip = 8000 * 2.5**4 / (8 * 200e9 * I2)  # wL^4/(8EI)

print("\n" + "=" * 60)
print("Case 2: Cantilever Beam, UDL w=-8kN/m")
print("=" * 60)

print(f"\nReactions: {r2.reactions}")
print(f"V_max: {r2.max_shear:.1f} (should be 20000 N)")

# Check shear at fixed end and tip
V_fixed = r2.shear[1]  # just to the right of x=0
V_tip = r2.shear[-1]
print(f"V@0+: {V_fixed:.1f} (should be +20000)")
print(f"V@L: {V_tip:.1f} (should be 0)")

# Critical: moment at fixed end
print(f"M@x=0: {r2.moment[0]:.1f} (should be {M2_fixed:.1f} at fixed end)")
print(f"M@x=L: {r2.moment[-1]:.1f} (should be 0 at free end)")
print(f"M_max: {r2.max_moment:.1f} (should be ~{M2_fixed:.1f})")

if abs(r2.moment[0]) < M2_fixed * 0.1:
    errors.append(f"CASE2: M@fixed_end should be ~{M2_fixed}, got {r2.moment[0]:.1f} — moment reaction not included in BMD!")

print(f"\nv_max: {r2.max_deflection*1e3:.3f} mm (should be {v2_tip*1e3:.3f} mm)")
print(f"v@x=0: {r2.deflection[0]*1e6:.2f} μm (should be 0 at fixed end)")
print(f"v@x=L: {r2.deflection[-1]*1e3:.3f} mm (max deflection at tip)")

if abs(r2.deflection[0]) > 1e-9:
    errors.append(f"CASE2: v@fixed_end should be 0")

# ===========================================================================
# Case 3: Simply-supported, uniform distributed load
# ===========================================================================
beam3 = BeamModel(
    length=5.0, beam_type=BeamType.SIMPLY_SUPPORTED,
    supports=[{"type": "pinned", "position": 0.0}, {"type": "roller", "position": 5.0}],
    point_loads=[],
    distributed_loads=[{"start": 0.0, "end": 5.0, "value": -10000}],
    cross_section={"type": CrossSection.RECTANGLE, "width": 0.2, "height": 0.4},
    material={"E": 200e9, "nu": 0.3, "yield_stress": 250e6},
)
r3 = analyzer.analyze(beam3)

I3 = 0.2 * 0.4**3 / 12
# M_max = wL^2/8 = 10000 * 25 / 8 = 31250
M3_max = 10000 * 25 / 8
# v_max = 5wL^4/(384EI)
v3_max = 5 * 10000 * 5.0**4 / (384 * 200e9 * I3)

print("\n" + "=" * 60)
print("Case 3: Simply-Supported Beam, UDL w=-10kN/m")
print("=" * 60)

print(f"\nReactions: {r3.reactions}")
print(f"V_max: {r3.max_shear:.1f} (should be 25000 N)")
print(f"V@x=0+: {r3.shear[1]:.1f} (should be +25000)")
print(f"V@x=L/2: {r3.shear[len(r3.shear)//2]:.1f} (should be 0)")
print(f"V@x=L-: {r3.shear[-2]:.1f} (should be -25000)")
print(f"\nM_max: {r3.max_moment:.1f} (should be {M3_max:.1f} N·m)")
print(f"M@x=0: {r3.moment[0]:.1f} (should be 0)")
print(f"M@x=L: {r3.moment[-1]:.1f} (should be 0)")
print(f"\nv_max: {r3.max_deflection*1e3:.3f} mm (should be {v3_max*1e3:.3f} mm)")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} BUG(S) FOUND:")
    for e in errors:
        print(f"  → {e}")
else:
    print("✅ All checks passed!")
print("=" * 60)
