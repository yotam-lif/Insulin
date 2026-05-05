# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase simulating **microscale insulin chemoreception** in a cylinder-in-cylinder geometry. Insulin diffuses through an annular region (capillary at r=a to cell wall at r=R) and binds to discrete receptor patches. The project validates closed-form analytical solutions against Monte Carlo (random walk) stochastic simulations.

All units: lengths in µm, time in s, concentration in particles/µm³.

## Running Scripts

Scripts live in `code/plotting/` and are run from the `code/` directory. Each is self-contained — no install step needed (they add `code/` to `sys.path` themselves):

```bash
cd code/
python plotting/rate_vs_radius.py        # Validates k = 2πHD/ln(R/a)
python plotting/absorbing_patches.py     # Tests Berg-Purcell limit vs patch count
python plotting/kinetics_ka.py           # Association rate sweep
python plotting/kinetics_kb.py           # Dissociation rate sweep
python plotting/kinetics_kc.py           # Internalization rate sweep
python plotting/transient.py             # Transient survival fraction
python plotting/concentration_profile.py # Equilibration to steady state
python plotting/equation_plots.py        # Analytical-only: μ vs coverage, bound% vs C0
```

Each script uses `ProcessPoolExecutor` with 8 workers by default (`MAX_WORKERS` constant), saves PNGs to `code/figures/` at 300 dpi, and prints numerical results to stdout.

## Dependencies

```
numpy, scipy (special, optimize, integrate), matplotlib
```

Python 3.14 virtualenv at `.venv/`. Activate with `source .venv/bin/activate` or just invoke `.venv/bin/python`.

## Architecture

### Data Flow

All parameters flow through a single `@dataclass`:

```
SimulationParams (classes/params.py)
    ↓
    ├── SteadyStateSimulation (stochastic_sim/steady_state.py)  → SteadyStateResult
    ├── TransientSimulation   (stochastic_sim/transient.py)     → dict {t, N_alive, frac}
    ├── AnalyticalModel       (analytical/steady_state.py)      → μ, flux, C(r), ...
    └── TransientAnalytical   (analytical/transient.py)         → survival_fraction(t)
```

Experiment scripts in `plotting/` call picklable worker functions in `stochastic_sim/workers.py`, which accept flat tuples (for `ProcessPoolExecutor`) and return flat tuples.

### Monte Carlo Step (per timestep, in both steady-state and transient)

1. Diffuse: `X += sqrt(2D·dt) · randn`
2. Detect outer-wall hits: `geometry.segment_hits_cylinder()` — quadratic ray-cylinder intersection
3. Bind/reflect: `patches.is_absorbing()` → absorb or `boundaries.reflect_outer()`
4. Release captured: `patches.release_captured()` — kb releases to outer bulk, kc to inner bulk (respawn near r=a)
5. Respawn absorbed (steady-state only): maintain constant N by placing new particles near r=a
6. Wrap z: `boundaries.wrap_periodic_z()` — periodic BC

### Key Design Decisions

**SimulationParams is immutable by convention** — passed by value, never mutated inside functions. Add fields there when extending the parameter space.

**PatchSet spatial hashing** (`stochastic_sim/patches.py`): Patches on the unrolled surface (R·θ, z) use a 2D grid for O(1) hit-to-patch lookup. Placement uses Random Sequential Adsorption (RSA).

**Erban-Chapman binding**: Finite `ka` uses probability `p = 1 − exp(−ka · dt / (√(2πD·dt) · patch_area))` rather than perfect absorption, so `ka=inf` means perfectly absorbing.

**C0 extrapolation**: Steady-state simulation estimates `C(r=a)` by linear fit through two thin shells near the inner wall (controlled by `eps_c0` and `eps_c0_2` in `SimulationParams`), since particles can't sit at r=a.

**Eigenfunction expansion** (`analytical/transient.py`): Cylindrical annulus eigenfunctions are φ(r) = J₀(λr) + β·Y₀(λr), with β = −J₁(λa)/Y₁(λa). Eigenvalues found via `scipy.optimize.brentq`.

### Physical Parameters

| Parameter | Symbol | Default range | Meaning |
|-----------|--------|---------------|---------|
| `a` | inner radius | ~3–5 µm | capillary radius |
| `R` | outer radius | ~10–20 µm | cell boundary |
| `H` | height | ~10 µm | periodic unit cell |
| `D` | diffusion | ~100 µm²/s | insulin diffusivity |
| `ka` | association | µm³/s or inf | binding rate; inf = perfect absorber |
| `kb` | dissociation | s⁻¹ | release back to outer bulk |
| `kc` | internalization | s⁻¹ or inf | transport to inner bulk |

**Berg-Purcell limit**: kb=0, kc=inf, ka=inf. Uptake efficiency: μ = Ns / (Ns + πH / (2·ln(R/a))).

**Fully absorbing wall**: μ = 1, k = 2πHD / ln(R/a).