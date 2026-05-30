# Policy Health Monitor (PHM)

Policy Health Monitor is the runtime reliability layer for learned robot policies: it
consumes a policy's internals (embeddings, actions), perception, and node and sensor
health, then emits a single arbitrated signal (`OK`, `DEGRADED`, `INTERVENE`, `STOP`),
each with a human-readable reason and a recommended action, so a learned controller can
be supervised in the loop. The differentiating component is an out-of-distribution
detector that runs on policy internals and fires before behavior collapses, consolidating
the detector, hysteresis, and metrics work from three prior projects (HELIX, BlackBoxRS,
Phantom-Braking) into one runtime-assurance stack.

The binding design and build contract lives in
[docs/superpowers/specs/2026-05-29-policy-health-monitor-design.md](docs/superpowers/specs/2026-05-29-policy-health-monitor-design.md).

## Packages (v0 foundation)

- `phm_msgs`: ROS 2 message package (`PolicyHealthStatus`, `PolicyEmbedding`, `DetectorVerdict`).
- `phm_core`: pure-Python detector logic (Detector ABC, Hysteresis, calibration, severity), unit-tested without a ROS graph.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install numpy==1.26.4 pytest ruff
.venv/bin/python -m pytest src/phm_core -q
.venv/bin/ruff check src/phm_core
```

`phm_core` carries no ROS dependency so its tests run fast and without `rclpy`. The ROS 2
nodes are thin wrappers over `phm_core`.

## License

MIT, see [LICENSE](LICENSE).
