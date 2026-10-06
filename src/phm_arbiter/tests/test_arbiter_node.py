"""ArbiterNode callbacks against slotted messages, as rclpy delivers them.

Regression: ``_verdict_callback`` used to set ``msg._recv_time`` on the received
DetectorVerdict. Generated rclpy message classes define ``__slots__``, so on a
live graph (first seen on the onboard computer of a Unitree GO2) the first
verdict raised AttributeError and killed the arbiter: /phm/health was never
published. The pure-Python core tests could not see this because they pass
plain objects. Here the fake DetectorVerdict is slotted like the real one.

rclpy and phm_msgs are not installed in the pure-Python test venv, so both are
faked through sys.modules (same approach as phm_detectors/tests/test_node_smoke.py).
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


class _Param:
    def __init__(self, value: float) -> None:
        self._v = value

    def get_parameter_value(self):
        return types.SimpleNamespace(double_value=float(self._v))


class _Logger:
    def info(self, *a, **k) -> None:
        pass

    def warning(self, *a, **k) -> None:
        pass


class _Publisher:
    def __init__(self) -> None:
        self.sent: list = []

    def publish(self, msg) -> None:
        self.sent.append(msg)


class _FakeNode:
    def __init__(self, name: str) -> None:
        self._params: dict = {}
        self.publisher = _Publisher()
        self.subscription_cb = None

    def declare_parameter(self, name, default, descriptor=None):
        self._params[name] = _Param(default)

    def get_parameter(self, name):
        return self._params[name]

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subscription_cb = cb
        return object()

    def create_publisher(self, msg_type, topic, qos):
        return self.publisher

    def create_timer(self, period, cb):
        return object()

    def get_logger(self):
        return _Logger()

    def get_clock(self):
        stamp = types.SimpleNamespace(sec=0, nanosec=0)
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: stamp))


class _SlottedVerdict:
    """Mirrors a generated rclpy message: no attribute outside __slots__."""

    __slots__ = ("header", "source", "score", "violating", "reason", "suggested_action")

    def __init__(self, source: str, score: float, violating: bool, action: int = 0) -> None:
        self.header = None
        self.source = source
        self.score = score
        self.violating = violating
        self.reason = f"{source} test"
        self.suggested_action = action


class _HealthStatus:
    def __init__(self) -> None:
        self.header = types.SimpleNamespace(stamp=None)
        self.state = None
        self.score = None
        self.reason = None
        self.source = None
        self.suggested_action = None


@pytest.fixture
def arbiter_node(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode
    qos_mod = types.ModuleType("rclpy.qos")
    for name in ("DurabilityPolicy", "HistoryPolicy", "ReliabilityPolicy"):
        setattr(qos_mod, name, types.SimpleNamespace(
            KEEP_LAST=0, RELIABLE=0, VOLATILE=0, TRANSIENT_LOCAL=0))
    qos_mod.QoSProfile = lambda **k: k
    rclpy.node, rclpy.qos = node_mod, qos_mod
    msgs = types.ModuleType("phm_msgs.msg")
    msgs.DetectorVerdict = _SlottedVerdict
    msgs.PolicyHealthStatus = _HealthStatus
    phm_msgs = types.ModuleType("phm_msgs")
    phm_msgs.msg = msgs
    for name, mod in {
        "rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos_mod,
        "phm_msgs": phm_msgs, "phm_msgs.msg": msgs,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "phm_arbiter.arbiter_node", raising=False)
    module = importlib.import_module("phm_arbiter.arbiter_node")
    return module.ArbiterNode()


def test_slotted_verdict_is_accepted(arbiter_node) -> None:
    with pytest.raises(AttributeError):
        _SlottedVerdict("x", 0.0, False)._recv_time = 1.0  # the fake really is slotted
    arbiter_node.subscription_cb(_SlottedVerdict("phm_ood_cpp", 0.0, False))


def test_live_verdict_reaches_health_output(arbiter_node) -> None:
    arbiter_node.subscription_cb(_SlottedVerdict("freq:/lowstate", 0.0, False))
    arbiter_node.subscription_cb(_SlottedVerdict("phm_ood_cpp", 0.9, True, action=3))
    arbiter_node._timer_callback()
    out = arbiter_node.publisher.sent[-1]
    assert out.state == 3
    assert out.source == "phm_ood_cpp"


def test_healthy_verdicts_publish_ok(arbiter_node) -> None:
    arbiter_node.subscription_cb(_SlottedVerdict("freq:/lowstate", 0.0, False))
    arbiter_node._timer_callback()
    assert arbiter_node.publisher.sent[-1].state == 0
