# Track A review punch list (from adversarial review, 2026-05-29)


## ros2-style
Verdict: BLOCKED. 2 blockers and 6 major findings must be resolved before merge. The critical issues are: (1) a confirmed DurabilityPolicy mismatch between the arbiter's TRANSIENT_LOCAL /phm/health publisher and the recovery node's implicit VOLATILE subscriber, which will silently drop all health messages to the recovery node at runtime; (2) a placeholder subscription type (`serialize_message.__class__` = `<class 'function'>`) in phm_detectors_node that will crash on any configured freq_topics or dead_topics; (3) the OodCore DEGRADED-band action bug that returns ACTION_NONE where the spec, docstring, and severity.py all require ACTION_LOG_ONLY; and (4) five of seven packages are excluded from CI build/test, so these failures would not be caught automatically. The .msg definitions match the spec exactly. The single-publisher invariant on /phm/health holds. Lifecycle callbacks in phm_ood are all five present and symmetric. Logging in callbacks is throttled throughout. Pure-Python core separation is clean and well-tested.

- [blocker] /home/yusuf/Projects/policy-health-monitor/src/phm_recovery/phm_recovery/recovery_node.py:48-51
  ISSUE: QoS mismatch on /phm/health: arbiter publishes with DurabilityPolicy.TRANSIENT_LOCAL (arbiter_node.py:77-81) but the recovery subscriber's _HEALTH_QOS omits DurabilityPolicy, which defaults to VOLATILE. In ROS2/DDS, a TRANSIENT_LOCAL publisher paired with a VOLATILE subscriber causes QoS incompatibility: the subscriber receives no messages, silently.
  FIX: Add `durability=DurabilityPolicy.TRANSIENT_LOCAL` to _HEALTH_QOS in recovery_node.py, and import DurabilityPolicy. The subscriber must match the publisher durability to receive late-joiner semantics.

- [blocker] /home/yusuf/Projects/policy-health-monitor/src/phm_detectors/phm_detectors/phm_detectors_node.py:161-165
  ISSUE: Broken subscription type and bare integer QoS. The create_subscription call passes `rclpy.serialization.serialize_message.__class__` (which resolves to `<class 'function'>`) as the message type. This will raise at runtime because rclpy requires a rosidl-generated type. Additionally, the depth argument is the bare integer 10, which violates the explicit-QoS rule.
  FIX: The subscription should use `AnyMsg` from `rclpy.serialization` or, better, route each monitored topic through a type-inferred subscriber using `wait_for_message` or topic introspection. Replace the bare `10` with a named QoSProfile. The current placeholder will crash the node on startup whenever any freq_topics or dead_topics are configured.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/_core.py:283-292
  ISSUE: OodCore._make_verdict assigns ACTION_NONE for the entire sub-INTERVENE range including the DEGRADED band (score in [0.25, 0.50)). The docstring at line 238 says 'DEGRADED/LOG_ONLY' for that band, and severity.py maps that band to ACTION_LOG_ONLY. The code instead returns ACTION_NONE. A confirmed OOD verdict in the DEGRADED band reaches the arbiter with `violating=True` but `suggested_action=0` (NONE), suppressing even a log-only advisory.
  FIX: Split the else branch: `elif score >= DEGRADED_THRESHOLD: action = ACTION_LOG_ONLY` then `else: action = ACTION_NONE`. Import ACTION_LOG_ONLY from phm_core.detector.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/_core.py:291
  ISSUE: When `fired=True` and score is in [0.0, 0.25) (i.e. below the DEGRADED band but raw_violating=True), the function returns `violating=True` with a reason annotated `[ok]`. The arbiter's _score_to_state computes STATE_OK for this score. A STATE_OK candidate with violating=True enters the worst-wins pool and can produce a STATE_OK result with a non-zero score, contradicting the spec invariant that 'State = OK if none violating'.
  FIX: If post-hysteresis score < DEGRADED_THRESHOLD, return violating=False with reason including '(below severity floor)'. A zero-severity OOD should be treated as a pass, not a confirmed violation.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_sim/phm_sim/embedding_publisher_node.py:38,61-108
  ISSUE: phm_sim imports `rcl_interfaces.msg.ParameterDescriptor` at module level (line 38) and uses it throughout the constructor, but phm_sim/package.xml declares no dependency on `rcl_interfaces`. Under a clean colcon install this import will fail when the overlay is not pre-loaded.
  FIX: Add `<depend>rcl_interfaces</depend>` to phm_sim/package.xml.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_detectors/phm_detectors/phm_detectors_node.py:34
  ISSUE: phm_detectors_node.py imports `rcl_interfaces.msg.ParameterDescriptor` at line 34 and uses it in _declare_params, but phm_detectors/package.xml has no `rcl_interfaces` dependency declared.
  FIX: Add `<depend>rcl_interfaces</depend>` to phm_detectors/package.xml.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/package.xml:20
  ISSUE: `<depend>python_cmake_module</depend>` is declared in phm_ood/package.xml. python_cmake_module is a C++ build support package (provides CMake macros for mixing Python/C++ in ament_cmake packages). phm_ood is a pure ament_python package with no CMakeLists.txt. This dep is incorrect and may confuse dependency resolvers.
  FIX: Remove `<depend>python_cmake_module</depend>` from phm_ood/package.xml.

- [major] /home/yusuf/Projects/policy-health-monitor/.github/workflows/ci.yml:37,43
  ISSUE: The colcon CI job only builds and tests `phm_msgs` and `phm_core`. The other five packages (phm_arbiter, phm_ood, phm_detectors, phm_recovery, phm_sim) are never built or tested in CI, so build breakage in those packages will not be caught before merge.
  FIX: Extend the `--packages-select` list in both `colcon build` and `colcon test` steps to include all packages, or use `colcon build` without `--packages-select` to build the full workspace.

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/node.py:28
  ISSUE: phm_ood/package.xml declares no numpy dependency (neither `python3-numpy` nor `numpy` as an exec_depend), yet node.py imports numpy at module level (line 28). The dep is satisfied transitively through phm_core in a colcon overlay but is not declared for standalone installs or rosdep.
  FIX: Add `<exec_depend>python3-numpy</exec_depend>` to phm_ood/package.xml (matching the pattern used in phm_sim/package.xml line 25).

- [major] /home/yusuf/Projects/policy-health-monitor/src/phm_arbiter/package.xml
  ISSUE: phm_arbiter/package.xml declares no `<buildtool_depend>ament_python</buildtool_depend>`. All other ament_python packages in this workspace (phm_detectors, phm_recovery, phm_sim) declare it. Without this entry, colcon may skip Python-specific install steps on some Humble configurations.
  FIX: Add `<buildtool_depend>ament_python</buildtool_depend>` to phm_arbiter/package.xml.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_arbiter/phm_arbiter/arbiter_node.py:70-73
  ISSUE: The verdict_qos QoSProfile for the /phm/verdicts subscription omits HistoryPolicy and DurabilityPolicy, relying on defaults (KEEP_LAST + VOLATILE). While the runtime defaults happen to be correct, the explicit-QoS rule requires all profiles to be fully declared so intent is unambiguous and any future default change in rclpy does not silently alter behavior.
  FIX: Add `history=HistoryPolicy.KEEP_LAST` and `durability=DurabilityPolicy.VOLATILE` to verdict_qos, and add `HistoryPolicy` to the rclpy.qos import.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_recovery/phm_recovery/recovery_node.py:48-50,55-58
  ISSUE: _HEALTH_QOS and _CMD_VEL_QOS both omit HistoryPolicy and DurabilityPolicy (apart from the DurabilityPolicy mismatch blocker already noted). Explicit declaration of all four QoS fields is required by the review criteria.
  FIX: Add `history=HistoryPolicy.KEEP_LAST` and (once the blocker is fixed) `durability=DurabilityPolicy.TRANSIENT_LOCAL` to _HEALTH_QOS; add `history=HistoryPolicy.KEEP_LAST` and `durability=DurabilityPolicy.VOLATILE` to _CMD_VEL_QOS. Import HistoryPolicy.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_detectors/phm_detectors/phm_detectors_node.py:56-60
  ISSUE: _VERDICT_QOS in phm_detectors_node.py omits HistoryPolicy and DurabilityPolicy. Same explicit-QoS rule violation as above.
  FIX: Add `history=QoSHistoryPolicy.KEEP_LAST` and `durability=QoSDurabilityPolicy.VOLATILE` to _VERDICT_QOS.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_arbiter/package.xml:12
  ISSUE: `<depend>std_msgs</depend>` is declared in phm_arbiter/package.xml but std_msgs is never imported in any phm_arbiter Python file. The dependency is satisfied transitively through phm_msgs. This is an orphan dependency.
  FIX: Remove `<depend>std_msgs</depend>` from phm_arbiter/package.xml.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/_core.py:283-292
  ISSUE: When score >= STOP_THRESHOLD (0.80) and fired=True, OodCore returns ACTION_HOLD (not ACTION_STOP_AND_HOLD). The comment says 'arbiter may escalate', but severity.py maps score >= 0.80 to ACTION_STOP_AND_HOLD. The arbiter does not escalate: it uses the verdict's suggested_action directly (clamped to allowlist). So STOP-level OOD from phm_ood never triggers STOP_AND_HOLD unless another detector escalates.
  FIX: Return ACTION_STOP_AND_HOLD when score >= STOP_THRESHOLD, consistent with severity.py and the spec's suggested_action table. If intentionally downgraded, add an explicit test case and comment explaining the design intent.

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_detectors/package.xml:22-23
  ISSUE: phm_detectors/package.xml declares both `<test_depend>ament_pytest</test_depend>` and `<test_depend>python3-pytest</test_depend>`. ament_pytest is a thin wrapper that depends on python3-pytest; declaring both is redundant. Additionally, ament_pytest as a rosdep key may not resolve on all Humble mirrors.
  FIX: Keep only `<test_depend>python3-pytest</test_depend>` (matching the pattern in phm_ood and phm_sim).

- [minor] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/node.py:275
  ISSUE: `self.get_logger().warn(...)` is used in _embedding_callback. `warn` is deprecated in rclpy Humble in favor of `warning`. While it currently works, it will produce a DeprecationWarning and is inconsistent with all other logging calls in the codebase.
  FIX: Replace `.warn(` with `.warning(` at node.py:275.

- [nit] /home/yusuf/Projects/policy-health-monitor/src/phm_arbiter/phm_arbiter/arbiter_node.py:106-108
  ISSUE: The startup `get_logger().info(...)` at line 106 fires once at construction (not in a per-tick callback) so it is correctly unthrottled. No action needed -- noted for completeness.
  FIX: No fix required.

- [nit] /home/yusuf/Projects/policy-health-monitor/src/phm_recovery/phm_recovery/recovery_node.py:118-121
  ISSUE: The `_sub_health = None` fallback path (ImportError guard) logs a warning but leaves the node running with no subscription. The node will spin indefinitely publishing nothing. A lifecycle-managed node would return FAILURE from on_configure; a plain node should at minimum log at ERROR level.
  FIX: Log at ERROR level and consider raising RuntimeError or calling rclpy.try_shutdown() to force an obvious failure rather than a silently-broken node.

- [nit] /home/yusuf/Projects/policy-health-monitor/src/phm_ood/phm_ood/_core.py:291
  ISSUE: The variable `state` is computed (lines 285, 288, 291) only to annotate the `reason` string via state_names dict (line 297). It is not stored on the returned DetectorVerdictData. The comment 'low severity OOD: log but do not intervene' implies a design intent that is not enforced by the action field. This is a latent clarity issue rather than a runtime bug at the current score bands.
  FIX: Consider removing the `state` variable entirely and replacing the reason annotation with a string computed from the action (e.g., 'log-only' vs 'hold') to avoid implying a state classification that is not propagated.


## skeptic-core
Verdict: Research-Only System. The calibration port is byte-faithful and the SafetyEnvelope correctly preserves the HELIX RESUME-cooldown-exemption, allowlist, and single-writer invariants. But the arbiter+recovery fusion path has real safety defects: a slightly-stale critical (STOP) verdict is silently downgraded to DEGRADED and clears the hold; the arbiter's REWIND-clamp-to-NONE collides with the recovery mapper so an INTERVENE+REWIND request becomes a no-op (no hold); a violating verdict with score below 0.25 maps to STATE_OK while still carrying score/source (the worst-wins map is NOT total over the violating x score space); NaN scores propagate to the published health output and pick STOP; and the recovery node's HOLD actuation is decoupled from the SafetyEnvelope so cooldown suppression is meaningless for holds while never re-arming a dropped hold. None of these are caught by the (otherwise thorough) tests because the tests only exercise self-consistent inputs. Not deployable as a reliability layer until the stale-critical downgrade and the INTERVENE-without-action gaps are closed.

- [blocker] src/phm_arbiter/phm_arbiter/_core.py:119-129 (stale branch), confirmed by runtime trace
  ISSUE: A stale-but-violating CRITICAL verdict is downgraded to DEGRADED. Any verdict with age > staleness_sec is unconditionally synthesized as STATE_DEGRADED/score 0.25, BEFORE checking violating/score. A 1.1s-old score=0.95 STOP_AND_HOLD verdict (verified: arbitrate -> state=1 DEGRADED, score=0.25, action=LOG_ONLY) loses its severity. Worse, downstream the recovery HealthToActionMapper treats STATE_DEGRADED as 'clear hold' (recovery/_core.py:241-249), so a transient verdict-delivery hiccup on a genuinely critical fault RELEASES the safety hold and lets the robot move. This is the exact opposite of fail-safe: staleness on a critical channel should escalate, never de-escalate.
  FIX: Make staleness fusion worst-of-(stale-floor, original-severity): for a stale verdict, take max(STATE_DEGRADED, _score_to_state(v.score)) and max(_STALE_SCORE, v.score) when violating, keeping reason 'stale:<source>'. Equivalently, never let the stale synthesis lower a verdict's own state. Add a test: stale + violating + score>=0.80 must stay STOP (or at least never drop below the live severity).

- [blocker] src/phm_arbiter/_core.py:130-150 + 174-194 and src/phm_recovery/_core.py:219-239 (cross-module), confirmed by runtime trace
  ISSUE: The worst-wins state map is NOT total over the (violating, score) input space, and the gap is a silent safety hole. A verdict with violating=True but score<0.25 (verified: score=0.10 -> _score_to_state returns STATE_OK) produces an arbiter result with state=STATE_OK yet score=0.10, source set, and a populated reason. A detector that explicitly asserts 'I am violating' is rendered invisible (OK). The spec (3.3 step 2: 'State = max severity state across non-stale VIOLATING verdicts') implies any violating verdict must be at least DEGRADED. The current banding ignores the violating flag entirely once past the stale check and re-derives state purely from score.
  FIX: For a non-stale violating verdict, floor the state at STATE_DEGRADED: state = max(STATE_DEGRADED, _score_to_state(v.score)). A violating=True verdict must never resolve to STATE_OK. Add the missing test (violating=True, score=0.1 -> state>=DEGRADED).

- [major] src/phm_arbiter/_core.py:38-41,132-134 vs src/phm_recovery/_core.py:219-239, confirmed by runtime trace
  ISSUE: REWIND requests are silently dropped end-to-end. The arbiter ARBITER_ALLOWLIST excludes ACTION_REWIND and clamps it to ACTION_NONE (verified: INTERVENE-band verdict with ACTION_REWIND -> arbiter emits state=INTERVENE, suggested_action=NONE). The recovery mapper then hits the STATE_INTERVENE branch with suggested_action < ACTION_HOLD and returns ACTION_LOG_ONLY with hold_active=False (verified). So a detector asking for REWIND produces NEITHER a rewind NOR a hold: a no-op. The two modules each made a locally-defensible choice (arbiter: 'rewind is a recovery decision'; recovery: 'low action = no hold') that compose into a dropped safety action. There is no test covering the arbiter->recovery seam, so this is invisible.
  FIX: Decide ownership of REWIND explicitly. Either (a) keep REWIND out of the arbiter but have the recovery node derive REWIND from STATE+reason, or (b) make STATE_INTERVENE always activate at least a HOLD regardless of suggested_action (an INTERVENE state with no actuation is itself the bug). At minimum, in HealthToActionMapper, STATE_INTERVENE with suggested_action below HOLD should still hold, not LOG_ONLY+release. Add an integration test threading arbitrate() output into HealthToActionMapper for all (state, action) pairs.

- [major] src/phm_recovery/phm_recovery/recovery_node.py:171-189,204-206; SafetyEnvelope cooldown at _core.py:129-140
  ISSUE: The SafetyEnvelope cooldown is decoupled from actual hold actuation, making the cooldown both ineffective and unsafe in opposite directions. Zero-velocity is published whenever self._mapper.hold_active is True (_on_publish_tick), independent of the envelope. (1) When the envelope returns SUPPRESSED_COOLDOWN for a repeated HOLD (verified: second HOLD within 5s -> SUPPRESSED_COOLDOWN, publish=False, but mapper.hold_active stays True), the hold keeps publishing anyway, so cooldown does nothing for an ongoing hold. (2) Conversely the envelope mutates _last_action_time on the FIRST hold even though publishing is driven by the mapper, so the cooldown bookkeeping is shadow state with no effect on the safety output. The HELIX design tied _current_action to the envelope result (recovery_node.py:133-138: 'if not result.publish: return'); this port broke that coupling by giving the mapper an independent hold flag. Net: the envelope's cooldown gate is decorative on the hold path.
  FIX: Make hold actuation a function of the envelope result, mirroring HELIX: only set/keep the mapper hold when env.evaluate(...).publish is True; if SUPPRESSED_COOLDOWN on a re-assert, do not start a NEW hold but also do not clear an existing one. Or, simpler and safer for a continuous hold: exempt 'continue an active hold' from cooldown entirely (cooldown should damp NEW holds, not an ongoing one) and remove the dead bookkeeping. Add a test asserting publish behavior tracks the envelope on repeated INTERVENE/HOLD.

- [major] src/phm_arbiter/_core.py:163,187 (max key over score; clamp in _score_to_state), confirmed by runtime trace
  ISSUE: NaN score is undefined behavior that reaches the published /phm/health. A verdict with score=NaN and violating=True yields state=STATE_STOP with score=NaN (verified). NaN survives because max(0.0, min(1.0, nan)) does not sanitize NaN in CPython, and float(msg.score) in arbiter_node.py:139 happily forwards a NaN from a misbehaving detector. The published PolicyHealthStatus.score is float32 NaN; the worst-wins tie-break max(..., key=lambda c:(c.state,c.score)) is also non-deterministic under NaN. A single buggy/poisoned detector can pin the whole monitor to STOP-with-NaN or produce nondeterministic arbitration.
  FIX: Validate inputs at the trust boundary: in arbiter_node._timer_callback reject or zero non-finite scores (math.isnan/isinf) and clamp to [0,1] before building the view; in _score_to_state replace the min/max clamp with an explicit isnan->treat-as-worst-or-reject policy. Fail loudly (log + drop the verdict, or force a defined STOP with reason 'bad-score:<source>'), never forward NaN.

- [minor] src/phm_arbiter/_core.py:43-48
  ISSUE: Stale-score comment is stale relative to the code. The docstring block says 'A stale verdict is synthesized as DEGRADED, score 0.5' and discusses 0.5 being 'at the boundary of INTERVENE', but _STALE_SCORE is actually 0.25. The narrative and the constant disagree, which is exactly the kind of drift that causes a later editor to 'fix' the constant back to 0.5 and silently push stale verdicts into INTERVENE. Misleading comments on safety-banding constants are a maintenance hazard.
  FIX: Rewrite the comment to match 0.25 (DEGRADED band lower edge) or hoist the value from severity.DEGRADED_THRESHOLD so the constant cannot drift from the band definition. Remove the obsolete 0.5/INTERVENE prose.

- [minor] src/phm_recovery/_core.py:124-144 (evaluate) and recovery_node.py:159-165
  ISSUE: RESUME exemption is implemented twice with different semantics, and the inline-evaluate RESUME path is dead/misleading. The comment at _core.py:124-128 claims 'a raw ACTION_NONE also bypasses cooldown' as the RESUME guard, but evaluate() never special-cases ACTION_NONE for resume: ACTION_NONE simply isn't in the (HOLD, STOP_AND_HOLD) cooldown set, so it bypasses cooldown only incidentally, and it also is not in _ACTUATING_ACTIONS so publish=False. The real resume path is evaluate_resume(). The comment describes a guard that does not exist, which obscures the actual single-writer/RESUME invariant for the next reader. The invariant itself is preserved (evaluate_resume is cooldown-exempt and disabled-gated, matching HELIX:53), but the duplicated explanation is a correctness-reasoning hazard.
  FIX: Delete the misleading ACTION_NONE 'resume guard' comment in evaluate(); document that RESUME is solely via evaluate_resume(). Keep evaluate_resume as the only cooldown-exempt path. No behavior change needed; this is to keep the invariant auditable.

- [minor] src/phm_arbiter/arbiter_node.py:114,119 vs src/phm_recovery/recovery_node.py:212-213
  ISSUE: Inconsistent clocks across the two safety nodes. The arbiter computes staleness from time.monotonic() (good: immune to wall-clock jumps and not driven by /clock). The recovery node computes cooldown timing from self.get_clock().now() (ROS time), which under use_sim_time follows /clock and can jump, pause, or run backward in replay/sim, and under system time can step on NTP corrections. Cooldown and staleness windows are thus on different, independently-warpable time bases, so 'sim certifies logic' claims about timing are only as valid as the sim clock monotonicity. A backward /clock jump can make (now - last) negative and permanently wedge a cooldown.
  FIX: Pick one time base for all safety timing. For wall-clock damping use a monotonic source in both nodes; if ROS time is required for sim determinism, guard against non-monotonic deltas (clamp now-last to >=0, reset bookkeeping on detected backward jumps). Document which clock each window uses in the spec.

- [nit] src/phm_core/phm_core/calibration.py:25-87 (port) vs phantom-braking/src/e6_detector.py:16-53
  ISSUE: Calibration port is byte-faithful: rolling_spread window indexing (range(window, T+1), out[t-1], var over [t-window:t], axis=0 sum) is identical; calibrate_threshold NaN-mask and np.percentile direction (low percentile = low spread = OOD) match; loco_fpr per-fold arithmetic matches, with the only intended divergence being dict-keyed -> list/index-keyed folds (documented, and covered by test_loco_parity_against_inlined_source_math). No math drift found. One latent fragility inherited from the source: percentile is np.percentile's 0-100 convention (default 1.0 = 1st percentile), which is easy to confuse with a 0-1 fraction; a caller passing 0.99 expecting the 99th percentile would get the ~1st. Not a port defect, but worth a guard.
  FIX: No port change required. Optionally add an assert 0 <= percentile <= 100 (or document the 0-100 convention loudly at the call sites in benchmark/loco code) to prevent a fraction-vs-percent mixup downstream.
