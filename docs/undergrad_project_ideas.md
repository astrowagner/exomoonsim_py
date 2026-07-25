# Undergraduate project ideas for the astrometric exomoon framework

The `exomoonsim` framework (Papers I–III) is now a validated machine for simulating
astrometric exomoon detection and characterization. These are candidate directions
for a student project aimed at a first-author paper, ordered roughly by how well they
fit a "run parameter studies and make plots" workflow (limited new code) while still
being publishable.

## 1. Which real imaged planets are the best exomoon-hunting grounds? (recommended lead)

Apply the framework to the actual population of directly imaged planets — β Pic b, the
HR 8799 planets, 51 Eri b, PDS 70 b/c, AF Lep b, GJ 504 b, etc. — using their real
distances, planet masses, and separations, and produce a detectability map (moon mass
vs separation) for each at realistic precisions (GRAVITY / GRAVITY+ reach tens of µas
today; note what ELT/MICADO or long-baseline interferometry might reach).

- **Deliverable:** a ranked target list — which imaged planets admit a 0.1–1 M⊕ moon,
  and at what precision — plus per-system maps. Observers cite target-list papers.
- **Student work:** (a) build a literature parameter table (good training in reading
  the field); (b) run one `run_survey` per system and assemble the maps. Almost no new
  code.
- **Paper III twist:** imaged planets have very long orbital periods, so the star–planet
  position angle barely sweeps over a campaign. The inclination/ellipse recovery
  (Paper III) therefore only works for the shorter-period or wider-motion systems.
  Mapping *where mass alone is recoverable vs where inclination is too* is a genuine
  finding, not just a rerun.

## 2. Multiple-moon confusion and reliability maps

Flagged as future work in Paper III. Systematically vary two/three-moon architectures —
period ratios (including resonances like 2:1, 3:2), mass ratios, phase offsets — and
measure when iterative prewhitening cleanly recovers everything vs blends moons or
produces a spurious extra "moon."

- **Deliverable:** a confusion/reliability map complementing the completeness maps of
  Papers I–II, built on the false-positive machinery of Paper III.
- **Student work:** pure crank-turning on the existing multi-moon and false-positive
  code.

## 3. Observing-strategy optimization

Given a fixed budget of epochs, what schedule maximizes detection *and* inclination
recovery? Sweep cadence, campaign length, and epoch count. The new axis versus Paper II
is that inclination recovery needs the planet's position angle to move, so there is a
real tension between many closely-spaced epochs and a long baseline.

- **Deliverable:** recovery accuracy vs (campaign length / planet period); a practical
  scheduling recommendation.
- **Student work:** low code, clear plots.

## 4. Eccentric-moon bias (a bit more coding)

The projected-ellipse fit assumes a circular moon orbit. Quantify how moon eccentricity
biases the recovered mass/inclination/period, find the eccentricity where the circular
model breaks, and test whether adding a couple of harmonic terms to the linear fit fixes
it.

- **Deliverable:** bias curves vs eccentricity; a recommended model order.
- **Student work:** a small, well-contained code change (extra columns in the
  least-squares design matrix) plus crank-turning — a good stretch if he wants to grow
  his programming a little.

## Suggested staffing

Start with **#1** as the backbone paper; let **#2** or **#3** become a second results
section if there is momentum. A `run_targets.py` driver (reads a systems CSV of
distance, planet mass, separation, precision; emits a detectability map + summary table
per system) plus a starter CSV of real systems would be a low-friction on-ramp.
