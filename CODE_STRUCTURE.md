# Code Structure: Insulin Chemoreception in a Cylinder-Cylinder Model

## Overview

This project simulates **microscale insulin chemoreception** in a simplified tissue geometry. The physical system represents a blood capillary (inner cylinder, radius `a`) surrounded by an annular tissue region extending to the outer cell boundary (outer cylinder, radius `R`). Insulin molecules diffuse through this annular space and bind to discrete receptor patches on the outer cylinder surface.

The project combines:
- **Analytical theory** — closed-form solutions for steady-state flux, concentration profiles, and transient survival fractions.
- **Monte Carlo random walk (RW) simulations** — stochastic particle simulations that validate and extend the analytical results.

All lengths are in **micrometres (µm)**, time in **seconds (s)**, and concentrations in **particles/µm³**.

---

## Repository Layout

```
Insulin/
├── code/
│   ├── classes/            # Data structures and parameter containers
│   ├── stochastic_sim/     # Monte Carlo random walk simulation engine
│   ├── analytical/         # Closed-form analytical solutions
│   ├── plotting/           # Standalone runnable figure scripts
│   └── Figures/            # Generated PNG output files
├── latex/                  # Academic paper source (LaTeX)
│   ├── main.tex
│   ├── compile.sh
│   └── sections/
│       ├── 01_model_geometry.tex
│       ├── 02_analytical_framework.tex
│       ├── 03_random_walk.tex
│       ├── 04_physiological_constraints.tex
│       ├── 05_system_constraints.tex
│       ├── 06_conclusions.tex
│       ├── A_appendix.tex
│       └── references.tex
└── CODE_STRUCTURE.md       # This file
```

### Running a figure script

Each script in `plotting/` is self-contained and runnable directly from the `code/` directory:

```bash
cd code/
python plotting/rate_vs_radius.py
python plotting/kinetics_kb.py
# etc.
```

Each script adds `code/` to `sys.path` automatically, so the sibling packages (`classes`, `stochastic_sim`, `analytical`) are found without any installation step.

---

## Dependencies

### Python Standard Library
| Module | Used for |
|--------|----------|
| `math` | Scalar math (`log`, `inf`, `sqrt`) |
| `dataclasses` | `@dataclass` parameter containers |
| `concurrent.futures.ProcessPoolExecutor` | CPU-parallel parameter sweeps |
| `time` | RNG seeding |

### Third-Party
| Package | Used for |
|---------|----------|
| `numpy` | Arrays, vectorised math, random number generation |
| `scipy.special` | Bessel functions `j0, j1, y0, y1` (eigenvalue expansion) |
| `scipy.optimize` | Root finding `brentq` (eigenvalues), `curve_fit` (model fitting) |
| `scipy.integrate` | Numerical quadrature `quad` (mode integrals) |
| `matplotlib` | All plotting |

---

## `classes/` — Data Structures

### `classes/params.py` — `SimulationParams`

Single `@dataclass` holding every parameter needed by any simulation or analytical calculation. Passed by value to all functions and constructors.

```python
@dataclass
class SimulationParams:
    N: int                    # number of simulation particles (required)
    C0: float = 0.0           # target bulk concentration (molecules/µm³)
    D: float = 0.0            # diffusion coefficient (µm²/s)
    a: float = 0.0            # inner cylinder radius (µm)
    R: float = 0.0            # outer cylinder radius (µm)
    H: float = 0.0            # cylinder height (µm)
    dt: float = 0.0           # time step (s)
    T: float = 0.0            # total simulation time (s)
    t_relax: float = 1.0      # equilibration time before measurements (s)
    n_patches: int = 0        # number of receptor patches on outer cylinder
    patch_radius: float|None  # patch radius in the unrolled (R·θ, z) plane (µm)
    n_sites_per_patch: int = 1
    ka: float = math.inf      # association rate (µm³/s); inf = perfectly absorbing
    kb: float = 0.0           # dissociation rate -- release to outer bulk (s⁻¹)
    kc: float = math.inf      # internalization rate -- release to inner bulk (s⁻¹)
    eps_c0: float = 0.1       # shell thickness for C0 extrapolation near r=a (µm)
    eps_respawn: float = 0.1  # shell thickness for respawning near r=a (µm)
    eps_out: float = 0.1      # shell thickness for outer-wall concentration (µm)
```

---

## `stochastic_sim/` — Monte Carlo Simulation Engine

### `stochastic_sim/geometry.py`

Pure geometric utilities with no simulation state.

#### `rand_points_in_shell(n, a, R, H, eps, rng, where) → (n, 3) array`
Samples `n` uniformly distributed points inside a thin cylindrical shell.
- `where="inner"` → `r ∈ [a, a+eps]`
- `where="outer"` → `r ∈ [R-eps, R]`

Uses area-uniform radial sampling: `r = sqrt(r0² + u·(r1²−r0²))`, `u ~ Uniform(0,1)`.

#### `segment_hits_cylinder(X, dX, R, snap=True) → (hit_mask, t_hit, S)`
Ray-cylinder intersection for all displacement segments `X → X+dX` against the cylinder `r = R`. Solves the quadratic in `t ∈ [0,1]`:

```
(x + t·dx)² + (y + t·dy)² = R²
```

Returns the earliest forward root, the hit boolean mask, and the intersection point `S` (optionally snapped exactly onto `r=R`).

---

### `stochastic_sim/boundaries.py`

Boundary condition handlers applied after each diffusion step.

#### `reflect_inner(X, a) → X`
Specular reflection for particles that stepped inside `r < a`. Moves each offending particle outward by `2·(a−ρ)` along the radial direction.

#### `reflect_outer(S_hit, dX, t_hit, R) → X_new`
Specular reflection from `r = R` for particles that hit the wall but were not absorbed. Reflects the remaining displacement `(1−t_hit)·dX` off the outward normal at the hit point.

#### `wrap_periodic_z(X, H) → X`
Wraps z-coordinates into `[−H/2, H/2)` via modular arithmetic.

---

### `stochastic_sim/patches.py` — `PatchSet`

Manages all receptor patches on the outer cylinder surface `r = R`.

**Construction**: places `n_patches` circular discs (radius `patch_radius`) on the unrolled surface `(R·θ, z)`. Uses **Random Sequential Adsorption (RSA)** with periodic torus topology to enforce non-overlapping placement:

```
distance² = (R·Δθ)² + Δz²   with periodic wrapping
```

Builds a **2D spatial hash grid** (cell size = `patch_radius`) over the unrolled surface for O(1) average-case hit-to-patch lookup.

**Key methods**:

| Method | Description |
|--------|-------------|
| `patch_index_of_hits(S)` | For each hit point in `S`, returns the patch index it falls on (−1 = open wall) |
| `is_absorbing(S)` | Boolean mask: True if hit is on any patch |
| `select_binders(idx_hit, ...)` | Selects earliest-hitting particle per patch up to free receptor capacity; returns `(K,2)` array of `[particle_idx, patch_idx]` |
| `release_captured(idx_captured, dt, rng)` | Stochastic release: each bound particle releases with `p_rel = 1−exp(−(kb+kc)·dt)`; splits into outer (fraction `kb/(kb+kc)`) and inner (fraction `kc/(kb+kc)`) |

---

### `stochastic_sim/steady_state.py`

#### `SteadyStateResult` (dataclass)

| Field | Description |
|-------|-------------|
| `c0_total` | Naïve global concentration `N/V_shell` |
| `c0_measured` | Extrapolated concentration near `r=a` |
| `rate` | Absorption rate (particles/s) after relaxation |
| `rate_normalized` | `rate / c0_measured` (µm³/s) |
| `r_centers` | Radial bin centres (if `concentration_profile=True`) |
| `profile_times` | Snapshot times (if `concentration_profile=True`) |
| `concentrations` | `(n_times, n_bins)` radial profiles (if `concentration_profile=True`) |

#### `SteadyStateSimulation`

Steady-state Monte Carlo. Absorbed particles are immediately respawned near `r=a` to maintain a constant particle count. After `t_relax`, the absorption rate and background concentration are recorded.

**Algorithm per time step**:
1. **Diffuse**: free particles step by `dX ~ N(0, √(2D·dt))` in each Cartesian direction
2. **Detect outer hits**: `segment_hits_cylinder` finds which displacement segments cross `r=R`
3. **Bind or reflect**: hits on a patch are gated by Erban-Chapman probability (finite `ka`), then by receptor capacity; non-binding hits are specularly reflected
4. **Reflect inner**: any free particle that penetrated `r < a` is reflected
5. **Release**: bound particles release with `p_rel = 1−exp(−(kb+kc)·dt)`; inner releases increment the absorption counter and respawn near `r=a`; outer releases respawn near `r=R`
6. **Wrap z**: periodic boundary in z

**C0 extrapolation**: time-averaged densities in two thin shells near `r=a` are linearly extrapolated back to `r=a` to estimate the bulk concentration.

**Erban-Chapman binding probability** (finite `ka`):
```
κ       = ka / (π·s²)
p_bind  = (κ·√(π·dt/D)) / (1 + κ·√(π·dt/D)/2)
```

**Usage**:
```python
sim = SteadyStateSimulation(params, rng)
result = sim.run()                        # basic
result = sim.run(concentration_profile=True)  # with radial snapshots
```

---

### `stochastic_sim/transient.py` — `TransientSimulation`

Transient Monte Carlo. Particles are placed once; absorbed particles are removed permanently (no respawning). Records survival fraction `N(t)/N₀` at regular intervals for comparison with the analytical eigenfunction expansion.

**Usage**:
```python
sim = TransientSimulation(params, rng)
res = sim.run(record_stride=100)
# res["t"], res["frac_alive"], res["N_alive"], res["absorbed"]
```

---

### `stochastic_sim/workers.py`

Top-level picklable functions for `ProcessPoolExecutor`. Each worker accepts a flat argument tuple, constructs `SimulationParams`, runs one simulation, and returns scalar results.

| Worker | Input tuple | Returns |
|--------|-------------|---------|
| `worker_rate_vs_radius` | `(a, D, R, H, dt, T, N, seed)` | `(a, k_sim, k_theory)` |
| `worker_absorbing_patches` | `(n_patches, t_relax, D, R, H, dt, T, N, a, patch_radius, seed)` | `(n_patches, sim_ratio, analytical, coverage)` |
| `worker_kinetics_steady` | `(n_patches, t_relax, D, R, H, dt, T, C0, a, patch_radius, ka, kb, kc, seed)` | `(n_patches, sim_ratio, analytical)` |

`worker_kinetics_steady` internally calls `AnalyticalModel.n_total(C0)` to determine the correct total particle count for the target bulk concentration before running the simulation.

---

## `analytical/` — Closed-Form Solutions

### `analytical/steady_state.py` — `AnalyticalModel`

Constructed from a `SimulationParams` instance. Provides all steady-state analytical results.

**Derived constants** (computed on access):

| Property | Formula |
|----------|---------|
| `ka_star` | `4·D·s / (1 + 4·D·s/ka)` — effective association rate |
| `kd` | `(kb + kc) / ka_star` — dissociation constant |
| `k_max` | `2π·H·D / ln(R/a)` — fully-absorbing rate constant (µm³/s) |
| `sigma0` | `n_patches / (2π·R·H)` — receptor surface density |

**Key methods**:

| Method | Formula / description |
|--------|-----------------------|
| `mu(C0)` | Uptake efficiency `J/J_max ∈ [0,1]`. Berg-Purcell limit if `kb=0, kc=∞`; otherwise solves quadratic from flux balance (see below) |
| `concentration_at_r(r, C0)` | `C0·(1 − μ·ln(r/a)/ln(R/a))` |
| `concentration_at_R(C0)` | `C0·(1 − μ)` |
| `flux(C0)` | `μ · k_max · C0` (particles/s) |
| `bound_fraction(C0)` | `C(R) / (C(R) + Kd)` — Michaelis-Menten occupancy |
| `bound_percent(C0)` | Occupancy as percentage |
| `n_bulk(C0)` | Volume integral of `C(r)` over the annulus |
| `n_total(C0)` | `n_bulk + n_bound` |
| `c0_from_n_total(N)` | Bisection inverse: find `C0` given `N_total` |

Also exports `molar_to_molecules_per_um3(c_molar)`.

**The mu formula** (finite kinetics):
```
A = D·(C0 + Kd) / (R·ln(R/a)) + σ₀·kc
B = 4·(D·C0 / (R·ln(R/a))) · σ₀·kc
C = 2·D·C0 / (R·ln(R/a))

μ = (A − √(A²−B)) / C
```

**Berg-Purcell limit** (`kb=0, kc=∞`):
```
μ = N·s / (N·s + π·H / (2·ln(R/a)))
```

---

### `analytical/transient.py` — `TransientAnalytical`

Eigenfunction expansion for transient diffusion in the cylindrical annulus with:
- **Neumann BC** at `r=a` (reflecting inner wall, always)
- **Dirichlet** or **Robin** BC at `r=R`

**Eigenfunctions**: `φ(r) = J₀(λr) + β·Y₀(λr)` with `β = −J₁(λa)/Y₁(λa)`

**Eigenvalue solvers**:
- `eigenvalues_dirichlet()`: solves `J₀(λR)·Y₁(λa) − Y₀(λR)·J₁(λa) = 0`
- `eigenvalues_robin(k)`: solves Robin equation with permeability parameter `k`

Both use a sign-change scan over a fine grid followed by `scipy.optimize.brentq` refinement.

**Survival fraction**:
```
N(t)/N₀ = Σₙ Aₙ · Iₙ · exp(−D·λₙ²·t)
```
where `Aₙ` are expansion coefficients and `Iₙ = ∫φₙ dr`, computed via `scipy.integrate.quad`.

**Usage**:
```python
ana = TransientAnalytical(params, n_modes=75)
surv = ana.survival_fraction(t, bc="robin", k=k_eff, p0="uniform_volume")
```

---

### `analytical/model_fit.py`

Empirical curve fitting for `J/J_max` as a function of receptor count. Tries 11 functional forms (logarithmic, power law, Michaelis-Menten-like, logistic) and reports R² for each. Useful for identifying the scaling law.

**Key functions**:
- `fit_J_over_Jmax(n_patches, sim_ratios, r_patch)` — fits all models, optionally plots top-3
- `fit_bp_shape(n_patches, sim_ratios, r_patch)` — fits the specific Berg-Purcell form `x/(b+x)`

---

## `plotting/` — Figure Scripts

Each script is standalone and runnable with `python plotting/<script>.py` from the `code/` directory. All figures are saved to `code/Figures/` at 300 dpi.

| Script | Experiment | Output file |
|--------|-----------|-------------|
| `rate_vs_radius.py` | Sweeps `a ∈ [1, 5]` µm, fully absorbing wall. Validates `k = 2πHD/ln(R/a)`. | `Figures/rate_vs_radius.png` |
| `absorbing_patches.py` | Sweeps patch count 100–1200. Compares `J/J_max` with Berg-Purcell formula. | `Figures/absorbing_patches.png` |
| `concentration_profile.py` | Single run with radial snapshots. Shows equilibration of `C(r)` towards steady state. | `Figures/concentration_profile.png` |
| `transient.py` | Patchy vs fully-absorbing transient. Overlays 2 RW curves + 2 analytical (Robin/Dirichlet) curves. | `Figures/transient_survival.png` |
| `kinetics_kb.py` | Sweeps `kb ∈ {1,10,50,100}` × patch count. Shows effect of dissociation rate. | `Figures/kinetics_kb.png` |
| `kinetics_kc.py` | Sweeps `kc ∈ {0.1,...,5}` × patch count. Also plots BP limit. | `Figures/kinetics_kc.png` |
| `kinetics_ka.py` | Sweeps `ka ∈ {0.1,1,2,100}` × patch count. Shows Erban-Chapman finite-ka effect. | `Figures/kinetics_ka.png` |
| `equation_plots.py` | Analytical only (no simulation). Plots `μ` vs coverage and bound % vs C0. | `Figures/mu_vs_coverage.png`, `Figures/bound_percent_vs_C0.png` |

All simulation scripts use `ProcessPoolExecutor` with `MAX_WORKERS = 8`. Adjust this constant at the top of each script to match available CPU cores.

---

## Key Physical Quantities and Formulas

### Diffusion relaxation time
```
τ_diff = (R − a)² / (π²·D)
```
Used to set `t_relax = k·τ_diff` (typically k = 15–30).

### Berg-Purcell uptake efficiency (perfectly absorbing patches)
```
μ_BP = N·s / (N·s + π·H / (2·ln(R/a)))
```
where `N` = patch count, `s` = patch radius (µm).

### General uptake efficiency (finite kinetics)
```
ka* = 4·D·s / (1 + 4·D·s / ka)
Kd  = (kb + kc) / ka*
σ₀  = N / (2π·R·H)

A = D·(C0 + Kd)/(R·ln(R/a)) + σ₀·kc
B = 4·(D·C0/(R·ln(R/a)))·σ₀·kc
C = 2·D·C0/(R·ln(R/a))

μ = (A − √(A²−B)) / C
```

### Steady-state concentration profile
```
C(r) = C0 · (1 − μ · ln(r/a) / ln(R/a))
```
Boundary values: `C(a) = C0` (inner, bulk), `C(R) = C0·(1−μ)` (outer, depleted).

### Total flux
```
J = μ · k_max · C0       k_max = 2π·H·D / ln(R/a)
```

### Receptor occupancy (Michaelis-Menten)
```
θ = C(R) / (C(R) + Kd)
```

### Erban-Chapman binding probability (finite `ka`)
```
κ      = ka / (π·s²)
p_bind = κ·√(π·dt/D) / (1 + κ·√(π·dt/D)/2)
```

---

## Typical Workflows

### 1. Validate the fully-absorbing rate formula
```
python plotting/rate_vs_radius.py
  → worker_rate_vs_radius (×7, parallel)
  → rate_vs_radius.png
```

### 2. Study Berg-Purcell patchy absorption
```
python plotting/absorbing_patches.py
  → worker_absorbing_patches (×7, parallel)
  → absorbing_patches.png
```

### 3. Kinetics parameter sweep
```
python plotting/kinetics_kb.py   # or kinetics_kc.py / kinetics_ka.py
  → for each kb: worker_kinetics_steady (×N_patches, parallel)
  → kinetics_kb.png
```

### 4. Transient survival analysis
```
python plotting/transient.py
  → TransientSimulation (patchy)
  → TransientSimulation (fully absorbing)
  → TransientAnalytical.survival_fraction (Robin BC)
  → TransientAnalytical.survival_fraction (Dirichlet BC)
  → transient_survival.png
```

### 5. Analytical-only equation plots
```
python plotting/equation_plots.py
  → AnalyticalModel.mu() and .bound_percent()
  → mu_vs_coverage.png, bound_percent_vs_C0.png
```
