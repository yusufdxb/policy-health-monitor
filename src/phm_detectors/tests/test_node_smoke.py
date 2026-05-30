"""Construction smoke test for PhmDetectorsNode (review decision 8).

The detectors node previously passed a bogus message type
(``rclpy.serialization.serialize_message.__class__`` == ``<class 'function'>``)
to ``create_subscription``, which crashed the node at startup whenever any
freq_topics or dead_topics were configured. This test asserts the node now
CONSTRUCTS without crashing when both lists are populated.

rclpy / rcl_interfaces / phm_msgs are not available in the pure-Python test
venv (tests must not import rclpy), so they are mocked via sys.modules with a
minimal fake Node base that records the create_* calls. The test verifies:

  1. The node constructs with freq_topics and dead_topics configured.
  2. No subscription is created with a non-class (e.g. a function) message type.
  3. Watched-topic subscriptions are deferred when the type is not on the graph
     (graph empty), and the node still builds.
  4. When a topic type IS resolvable on the graph, a subscription binds with a
     real message class and an explicit QoSProfile (not a bare int).
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

# Ensure phm_detectors is importable (mirrors conftest, explicit here for clarity).
_DET_ROOT = Path(__file__).resolve().parents[1]
if str(_DET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DET_ROOT))


# ---------------------------------------------------------------------------
# Minimal rclpy / rcl_interfaces / phm_msgs fakes
# ---------------------------------------------------------------------------


class _FakeLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class _FakeParamValue:
    def __init__(self, value):
        self._v = value

    @property
    def string_array_value(self):
        return list(self._v) if isinstance(self._v, (list, tuple)) else []

    @property
    def double_value(self):
        return float(self._v) if isinstance(self._v, (int, float)) else 0.0

    @property
    def integer_value(self):
        return int(self._v) if isinstance(self._v, (int, float)) else 0


class _FakeParam:
    def __init__(self, value):
        self._v = value

    def get_parameter_value(self):
        return _FakeParamValue(self._v)


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _FakeNode:
    """Records create_subscription calls so the test can inspect message types."""

    _graph: dict = {}  # topic -> [type_name]; class attr so the test can set it

    def __init__(self, name):
        self._name = name
        self._params: dict = {}
        self.subscriptions_made: list = []
        self.publishers_made: list = []
        self.timers_made: list = []

    def declare_parameter(self, name, default, descriptor=None):
        # The real node declares the array params with a None default; the
        # _build_node helper overwrites them with the test values afterward.
        self._params[name] = _FakeParam([] if default is None else default)

    def get_parameter(self, name):
        return self._params[name]

    def get_parameter_or(self, name, default=None):
        # Mirrors rclpy.get_parameter_or: returns the param if set, else a
        # NOT_SET-equivalent (here an empty fake param) so .string_array_value
        # yields [].
        return self._params.get(name, _FakeParam([]))

    def create_publisher(self, msg_type, topic, qos):
        self.publishers_made.append((msg_type, topic, qos))
        return object()

    def create_subscription(self, msg_type, topic, cb, qos):
        # Record EVERYTHING so the test can assert no bogus types slip through.
        self.subscriptions_made.append((msg_type, topic, qos))
        return object()

    def create_timer(self, period, cb):
        t = _FakeTimer()
        self.timers_made.append((period, cb))
        return t

    def get_logger(self):
        return _FakeLogger()

    def get_topic_names_and_types(self):
        return [(t, v) for t, v in self._graph.items()]


def _install_fakes(graph: dict | None = None):
    """Install fake rclpy/rcl_interfaces/phm_msgs into sys.modules and return the
    freshly (re)imported node module.
    """
    _FakeNode._graph = graph or {}

    # rclpy package + submodules.
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _FakeNode

    rclpy_qos = types.ModuleType("rclpy.qos")

    class _QoSProfile:
        def __init__(self, **kw):
            self.kw = kw

    class _Enum:
        KEEP_LAST = "KEEP_LAST"
        RELIABLE = "RELIABLE"
        BEST_EFFORT = "BEST_EFFORT"
        VOLATILE = "VOLATILE"
        TRANSIENT_LOCAL = "TRANSIENT_LOCAL"

    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.QoSReliabilityPolicy = _Enum
    rclpy_qos.QoSHistoryPolicy = _Enum
    rclpy_qos.QoSDurabilityPolicy = _Enum

    rclpy.node = rclpy_node
    rclpy.qos = rclpy_qos

    # rcl_interfaces.msg.ParameterDescriptor
    rcl_interfaces = types.ModuleType("rcl_interfaces")
    rcl_msg = types.ModuleType("rcl_interfaces.msg")

    class _ParameterDescriptor:
        def __init__(self, description="", type=None, **kw):  # noqa: A002
            self.description = description
            self.type = type

    class _ParameterType:
        PARAMETER_STRING_ARRAY = 9  # value mirrors rcl_interfaces ParameterType

    rcl_msg.ParameterDescriptor = _ParameterDescriptor
    rcl_msg.ParameterType = _ParameterType
    rcl_interfaces.msg = rcl_msg

    # phm_msgs.msg.DetectorVerdict (a real class so it is a valid msg type).
    phm_msgs = types.ModuleType("phm_msgs")
    phm_msgs_msg = types.ModuleType("phm_msgs.msg")

    class DetectorVerdict:  # a real class -> a valid create_publisher type
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None)
            self.source = ""
            self.score = 0.0
            self.violating = False
            self.reason = ""
            self.suggested_action = 0

    phm_msgs_msg.DetectorVerdict = DetectorVerdict
    phm_msgs.msg = phm_msgs_msg

    # rosidl_runtime_py.utilities.get_message (returns a real class).
    rosidl = types.ModuleType("rosidl_runtime_py")
    rosidl_utils = types.ModuleType("rosidl_runtime_py.utilities")

    class _StdString:  # stand-in resolved message class
        pass

    rosidl_utils.get_message = lambda name: _StdString
    rosidl.utilities = rosidl_utils

    mods = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "rcl_interfaces": rcl_interfaces,
        "rcl_interfaces.msg": rcl_msg,
        "phm_msgs": phm_msgs,
        "phm_msgs.msg": phm_msgs_msg,
        "rosidl_runtime_py": rosidl,
        "rosidl_runtime_py.utilities": rosidl_utils,
    }
    for k, v in mods.items():
        sys.modules[k] = v

    # Drop any cached real import then import fresh against the fakes.
    sys.modules.pop("phm_detectors.phm_detectors_node", None)
    return importlib.import_module("phm_detectors.phm_detectors_node")


@pytest.fixture
def cleanup_modules():
    saved = {k: sys.modules.get(k) for k in (
        "rclpy", "rclpy.node", "rclpy.qos", "rcl_interfaces", "rcl_interfaces.msg",
        "phm_msgs", "phm_msgs.msg", "rosidl_runtime_py", "rosidl_runtime_py.utilities",
        "phm_detectors.phm_detectors_node",
    )}
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    sys.modules.pop("phm_detectors.phm_detectors_node", None)


def _build_node(mod, freq_topics, dead_topics):
    """Construct PhmDetectorsNode with the given topic lists.

    The fake Node stores each declared parameter's default in ``_params``. We
    wrap ``_declare_params`` to overwrite the two list params with the desired
    test values right after the node declares its defaults, so the node reads
    our topic lists when it builds the adapters.
    """
    node_cls = mod.PhmDetectorsNode
    orig_declare = node_cls._declare_params

    def patched_declare(self):
        orig_declare(self)
        # Overwrite the list params with the test values (same fake param wrapper
        # the node's get_parameter(...).string_array_value path expects).
        self._params["freq_topics"] = _FakeParam(list(freq_topics))
        self._params["dead_topics"] = _FakeParam(list(dead_topics))

    node_cls._declare_params = patched_declare
    try:
        return node_cls()
    finally:
        node_cls._declare_params = orig_declare


def test_node_constructs_with_topics_configured(cleanup_modules):
    """The node builds with freq_topics and dead_topics set, no crash."""
    mod = _install_fakes(graph={})  # empty graph: watched types not resolvable yet
    node = _build_node(mod, freq_topics=["/scan"], dead_topics=["/odom"])
    assert node is not None
    # No subscription bound yet (types not on the graph), node still constructed.
    sub_topics = [t for (_mt, t, _q) in node.subscriptions_made]
    assert "/scan" not in sub_topics
    assert "/odom" not in sub_topics


def test_no_subscription_uses_a_function_as_message_type(cleanup_modules):
    """Regression: NO create_subscription is called with a non-class message type
    (the old bug passed a function, ``serialize_message.__class__``).
    """
    mod = _install_fakes(graph={"/scan": ["std_msgs/msg/String"]})
    node = _build_node(mod, freq_topics=["/scan"], dead_topics=[])
    for msg_type, topic, qos in node.subscriptions_made:
        assert inspect.isclass(msg_type), (
            f"subscription on {topic} used a non-class msg type: {msg_type!r}"
        )
        # QoS must be an explicit profile object, never a bare int.
        assert not isinstance(qos, int), f"bare int QoS on {topic}"


def test_resolvable_topic_binds_with_real_class_and_qos(cleanup_modules):
    """When the topic type IS on the graph, a subscription binds with a real
    message class and an explicit QoSProfile.
    """
    mod = _install_fakes(graph={"/scan": ["std_msgs/msg/String"]})
    node = _build_node(mod, freq_topics=["/scan"], dead_topics=[])
    bound = [(mt, t, q) for (mt, t, q) in node.subscriptions_made if t == "/scan"]
    assert len(bound) == 1, "expected exactly one /scan subscription"
    msg_type, _topic, qos = bound[0]
    assert inspect.isclass(msg_type)
    # An explicit QoS profile object (our fake QoSProfile), never a bare int.
    assert not isinstance(qos, int)
    assert hasattr(qos, "kw"), f"expected a QoSProfile, got {qos!r}"
