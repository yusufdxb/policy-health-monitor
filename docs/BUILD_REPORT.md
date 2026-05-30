# Policy Health Monitor: Track A Integration and Build Report

Date: 2026-05-30 (updated; original integration 2026-05-29)
Stage: Track A fix pass (adversarial-review punch list) re-verified on top of Integrate
Environment: mewtwo, ROS 2 Humble, Python 3.10.12, venv at `.venv` (numpy 1.26.4, pytest 9.0.3, ruff 0.15.10)

## Summary

| Gate | Result |
|---|---|
| `colcon build` (6 packages, clean) | PASS, exit 0, `Summary: 6 packages finished` |
| `pytest src -q` (pure-Python suite) | PASS, `255 passed` (was 179 at integrate, 184/4-fail at start of fix pass) |
| `ruff check src/` | PASS, `All checks passed!` |
| Overall | GREEN |

## Fix pass (2026-05-30): adversarial-review punch list

The fix pass implemented the LOCKED safety decisions from
`docs/REVIEW_PUNCHLIST.md` and re-verified all three gates. Test count rose from
179 (integrate) to 255 as new safety regression and seam-integration tests were
added (4 old arbiter tests that encoded the OLD unsafe stale-downgrade behavior
were rewritten to assert the new fail-safe contract):

- D1 staleness never de-escalates a violating verdict (arbiter `_core`).
- D2 violating floor + OOD severity-floor (arbiter `_core`, phm_ood `_core`).
- D3 OOD action banding LOG_ONLY / HOLD / STOP_AND_HOLD (phm_ood `_core`).
- D4 REWIND seam: arbiter pass-through + recovery INTERVENE-always-holds, with a
  cross-module arbitrate() -> HealthToActionMapper integration test.
- D5 cooldown couples to actuation; ongoing hold is cooldown-exempt (phm_recovery).
- D6 NaN / non-finite guard at the arbiter trust boundary.
- D7 explicit 4-policy QoS on every PHM endpoint; recovery `/phm/health`
  durability set to TRANSIENT_LOCAL to match the arbiter publisher.
- D8 phm_detectors broken subscription replaced with ROS-graph type resolution +
  explicit QoS; STRING_ARRAY params fixed so configured topics are accepted; node
  verified to construct live with `freq_topics=[/scan] dead_topics=[/odom]`.
- D9 package.xml dependency fixes + CI now builds and tests all 6 packages.

## 1. Environment

`.venv` was already created by the Scaffold stage and is complete. Verified:

```
Python 3.10.12
numpy 1.26.4
pytest 9.0.3
ruff 0.15.10
```

No repair was needed on the venv. `phm_core` is a pure-Python package that is not
pip-installed into the venv; the per-package `conftest.py` files inject
`src/phm_core` onto `sys.path` for test collection, so this is expected and tests
pass without an editable install.

## 2. colcon build

Command (run from the repo root after `source /opt/ros/humble/setup.bash`, with the
go2_ws overlay unset so it does not interfere):

```
colcon build
```

Final output tail (clean `rm -rf build install log && colcon build`, 2026-05-30 fix pass):

```
Starting >>> phm_msgs
Finished <<< phm_msgs [2.36s]
Starting >>> phm_arbiter
Starting >>> phm_detectors
Starting >>> phm_ood
Starting >>> phm_recovery
Starting >>> phm_sim
Finished <<< phm_sim [1.06s]
Finished <<< phm_detectors [1.07s]
Finished <<< phm_ood [1.07s]
Finished <<< phm_recovery [1.07s]
Finished <<< phm_arbiter [1.08s]

Summary: 6 packages finished [3.51s]
```

Exit code 0. No packages failed. All five rclpy node modules import after the
clean rebuild, and `phm_detectors` was verified to construct live with
`freq_topics=[/scan] dead_topics=[/odom]` configured (the exact condition that
crashed the old bogus-subscription code, D8).

### Generated message contract verified

After sourcing `install/setup.bash`, the three interfaces import and their fields and
enum constants match the spec (section 3.1) exactly:

```
PolicyHealthStatus constants: STATE 0 1 2 3 | ACTION 0 1 2 3 4
PolicyHealthStatus fields: header, state, score, reason, source, suggested_action
PolicyEmbedding fields:     header, embedding, dim, policy_id
DetectorVerdict fields:     header, source, score, violating, reason, suggested_action
```

### Node executables verified

`ros2 pkg executables` resolves a runnable node for every rclpy package:

```
phm_ood        -> phm_ood_node
phm_arbiter    -> phm_arbiter
phm_detectors  -> phm_detectors_node
phm_recovery   -> recovery_node
phm_sim        -> embedding_publisher
```

All five node modules import cleanly (phm_msgs from the install space, phm_core from
`src` on PYTHONPATH): `phm_ood.node`, `phm_arbiter.arbiter_node`,
`phm_detectors.phm_detectors_node`, `phm_recovery.recovery_node`,
`phm_sim.embedding_publisher_node`.

## 3. Integration fixes applied (no logic redesign)

Three integration breaks were found and fixed. None touched detector, arbiter,
recovery, or calibration logic.

1. colcon `setup.py` introspection crash on four ament_python packages
   (`phm_ood`, `phm_detectors`, `phm_recovery`, `phm_sim`).
   Each carried both a `setup.py` and a `pyproject.toml` with a `[project]` table.
   colcon's `colcon_python_setup_py` runs `setup.py --dry-run` and then
   `ast.literal_eval`s the metadata dict; the `[project]` table caused setuptools to
   inject non-literal reprs (e.g. `<SpecifierSet('>=3.10')>`) that are not
   `literal_eval`-able, aborting the build with `SyntaxError: invalid syntax`.
   Fix: stripped the `[build-system]` and `[project]`/`[tool.setuptools...]` tables
   from those four `pyproject.toml` files, keeping only `[tool.pytest.ini_options]`
   and `[tool.ruff]` tool config. Packaging metadata now comes solely from
   `setup.py` + `package.xml` (the ament_python convention). `phm_core` keeps its
   `[project]` table because it is a pure pip package, not a colcon package.

2. Console scripts installed to `bin/` instead of `lib/<pkg>/` for `phm_arbiter`
   and `phm_recovery`, so `ros2 run`/`ros2 pkg executables` could not find them.
   `phm_arbiter/setup.cfg` was missing the `[install] install_scripts=...` section
   and `phm_recovery` had no `setup.cfg` at all.
   Fix: added/completed both `setup.cfg` files with
   `[install] install_scripts=$base/lib/<pkg>` (matching the working packages
   `phm_ood`, `phm_detectors`, `phm_sim`).

3. Pure-Python test collection broke under the bare spec command `pytest src -q`.
   Two issues: (a) every package's `tests/__init__.py` collided across packages
   under pytest's default "prepend" import mode (`No module named tests.test_*`);
   (b) `phm_arbiter` had no `conftest.py` and `phm_sim`'s `conftest.py` injected
   `src` instead of its own package root, so `import phm_arbiter` / `import phm_sim`
   failed.
   Fix: added a repo-root `pytest.ini` with `addopts = --import-mode=importlib`
   (imports each test file under a unique path-derived name, removing the
   collision); added `src/phm_arbiter/conftest.py` and corrected
   `src/phm_sim/conftest.py` to inject the correct package root plus `phm_core`.
   No test bodies were modified.

## 4. pytest (full pure-Python suite)

Command:

```
.venv/bin/python -m pytest src -q
```

Output tail (after the 2026-05-30 fix pass):

```
........................................................................ [ 28%]
........................................................................ [ 56%]
........................................................................ [ 84%]
.......................................                                  [100%]
255 passed in 0.20s
```

Per-package breakdown (sums to 255):

```
phm_core       36 passed
phm_ood        22 passed   (was 17; +5 severity-floor / action-banding tests, D2/D3)
phm_arbiter    50 passed   (was 38; +stale/violating-floor/NaN safety + REWIND seam, D1/D2/D6)
phm_detectors  31 passed   (was 28; +3 construction smoke tests, D8)
phm_recovery   95 passed   (was 39; +cooldown-coupling + arbiter->recovery seam integration, D4/D5)
phm_sim        21 passed
```

No tests import rclpy: node logic lives in pure modules (`_core.py` / `_sim_core.py`
/ `phm_core`) and the rclpy nodes are thin wrappers, so the suite runs without a ROS
graph. Note: the run uses an empty `PYTHONPATH` to keep the sourced go2_ws/Humble
overlay out of `sys.path`; CI runs in a clean setup-python environment where that
overlay is already absent, so the bare command behaves identically there.

## 5. ruff

Command:

```
.venv/bin/ruff check src/
```

Output:

```
All checks passed!
```

No lint fixes were required.

## 6. Packages left red

None. All six colcon packages build and all six pure-Python test suites pass.

## 7. Honest caveats (not build blockers)

- `phm_core` is not installed into the colcon install space (it is a pure pip
  package). The rclpy nodes `import phm_core` at runtime, so a live `ros2 run` of
  `phm_ood`, `phm_detectors`, or `phm_recovery` needs `phm_core` on `PYTHONPATH`
  (via the venv, an editable install, or a future ament wrapper). This is a runtime
  deployment detail, not a build or test failure, and the spec gates (colcon build +
  pytest) are met. ASSUMPTION: a launch-time PYTHONPATH or editable install of
  phm_core is acceptable for v0; if the user wants phm_core in the install space, it
  should be repackaged as an ament_python package in a follow-up.
- This build certifies LOGIC only (sim-certifiable), never physics or hardware
  behavior, per spec section 2. No end-to-end ROS graph run with live publishers was
  exercised in this stage; node-module imports and message round-trips were verified.
