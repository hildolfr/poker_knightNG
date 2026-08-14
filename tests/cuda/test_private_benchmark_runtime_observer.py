from importlib import import_module
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]


def _load_benchmark_tool():
    path = ROOT / "tools/benchmark_equity.py"
    spec = importlib.util.spec_from_file_location("benchmark_equity_observer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_result(module, count):
    return module.MonteCarloResult(
        completed_trials=count,
        unique_wins=count,
        tie_by_other_winners=(0, 0, 0, 0, 0, 0),
        losses=0,
        equity_share_units=420 * count,
        hero_category_counts=(count, 0, 0, 0, 0, 0, 0, 0, 0),
        rejection_count=0,
    )


class FakeCP:
    __version__ = "14.1.1"
    uint8 = "uint8"
    uint32 = staticmethod(int)
    uint64 = staticmethod(int)

    def __init__(self):
        self.asarray_calls = []
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(deviceSynchronize=lambda: None),
        )

    def asarray(self, value, dtype=None):
        self.asarray_calls.append((value, dtype))
        return ("device", tuple(value), dtype)

    @staticmethod
    def empty(size, dtype=None):
        return {"size": size, "dtype": dtype}

    @staticmethod
    def asnumpy(value):
        return value


class Observer:
    def __init__(self):
        self.events = []
        self.copies = []
        self.actions = []
        self.finished = False

    def prepare_inputs(self, hero, board):
        self.actions.append("prepare_inputs")
        self.inputs = (("prepared-hero", hero), ("prepared-board", board))

    def copy_inputs(self):
        self.actions.append("copy_inputs")
        return self.inputs

    def boundary(self, stage, edge, ordinal):
        self.actions.append(f"{stage}:{edge}:{ordinal}")
        self.events.append((stage, edge, ordinal))

    def copy_to_host(self, final, ordinal):
        self.copies.append((final, ordinal))
        return final

    def finish(self):
        self.finished = True
        return {
            "h2d_gpu_ns": "11",
            "simulate_gpu_ns": "22",
            "reduction_gpu_ns": "33",
            "d2h_gpu_ns": "44",
        }


def test_private_benchmark_path_has_exact_stage_boundaries_and_same_result(monkeypatch):
    tool = _load_benchmark_tool()
    module = import_module("poker_knight_ng._cuda_runtime")
    cp = FakeCP()
    runtime = module.CupyDeterministicRuntime(batch_blocks=1, _cp=cp)
    monkeypatch.setattr(runtime, "_batch_capacity", lambda: 1)
    launches = []

    def simulate(grid, block, arguments):
        launches.append(("simulate", grid, block, arguments[7]))

    def reduce(grid, block, arguments):
        launches.append(("reduction", grid, block, arguments[1]))

    monkeypatch.setattr(runtime, "_kernels", lambda: (simulate, reduce))
    observer = Observer()
    validation_states = []

    def validate_after_fence(_value, *, requested_trials, **_kwargs):
        validation_states.append(observer.finished)
        return _valid_result(module, requested_trials)

    monkeypatch.setattr(
        module,
        "validated_aggregate",
        validate_after_fence,
    )

    result, profile = tool._run_private_stage(
        runtime,
        module,
        hero=(2, 9),
        board=(),
        opponents=1,
        key=(0, 0),
        count=300,
        observer=observer,
        planned_batch_blocks=1,
    )

    assert result == _valid_result(module, 300)
    assert validation_states == [True, True, True, True]
    assert profile == observer.finish()
    assert module.THREADS == 128
    assert cp.asarray_calls == []
    assert observer.actions[:4] == [
        "prepare_inputs",
        "h2d:start:0",
        "copy_inputs",
        "h2d:end:0",
    ]
    assert launches == [
        ("simulate", (1,), (128,), 128),
        ("reduction", (1,), (128,), 1),
        ("simulate", (1,), (128,), 128),
        ("reduction", (1,), (128,), 1),
        ("simulate", (1,), (128,), 44),
        ("reduction", (1,), (128,), 1),
    ]
    assert observer.events == [
        ("h2d", "start", 0),
        ("h2d", "end", 0),
        ("simulate", "start", 0),
        ("simulate", "end", 0),
        ("reduction", "start", 0),
        ("reduction", "end", 0),
        ("d2h", "start", 0),
        ("d2h", "end", 0),
        ("simulate", "start", 1),
        ("simulate", "end", 1),
        ("reduction", "start", 1),
        ("reduction", "end", 1),
        ("d2h", "start", 1),
        ("d2h", "end", 1),
        ("simulate", "start", 2),
        ("simulate", "end", 2),
        ("reduction", "start", 2),
        ("reduction", "end", 2),
        ("d2h", "start", 2),
        ("d2h", "end", 2),
    ]
    assert [ordinal for _final, ordinal in observer.copies] == [0, 1, 2]


class Event:
    def __init__(self, index):
        self.index = index
        self.records = []
        self.synchronized = False

    def record(self, stream):
        self.records.append(stream)

    def synchronize(self):
        self.synchronized = True


class DeviceArray:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.set_calls = []
        self.get_calls = []

    def set(self, source, *, stream):
        self.set_calls.append((tuple(source), stream))

    def get(self, *, out, stream, blocking):
        self.get_calls.append((out, stream, blocking))
        out[:] = bytes(len(out))


class EventCP:
    uint8 = "uint8"

    def __init__(self):
        self.stream = object()
        self.events = []
        self.arrays = []
        self.pinned = []

        def make_event():
            event = Event(len(self.events))
            self.events.append(event)
            return event

        def empty(size, dtype):
            array = DeviceArray(size, dtype)
            self.arrays.append(array)
            return array

        def pinned(size):
            value = bytearray(size)
            self.pinned.append(value)
            return value

        self.empty = empty
        self.cuda = SimpleNamespace(
            Event=make_event,
            get_current_stream=lambda: self.stream,
            get_elapsed_time=lambda start, end: float(end.index - start.index),
            alloc_pinned_memory=pinned,
        )


def test_cupy_event_observer_preallocates_async_copies_and_resolves_after_fence(monkeypatch):
    tool = _load_benchmark_tool()
    cp = EventCP()
    fake_numpy = SimpleNamespace(
        uint8="uint8",
        asarray=lambda values, dtype: tuple(values),
        frombuffer=lambda memory, dtype, count: memory,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    observer = tool.CupyStageObserver(cp, batches=2, aggregate_bytes=192)
    assert len(cp.events) == 14
    assert len(cp.pinned) == 2

    observer.prepare_inputs((2, 9), ())
    assert len(cp.arrays) == 2
    assert len(cp.pinned) == 4
    assert cp.pinned[2] == bytearray((2, 9))
    assert cp.pinned[3] == bytearray((0,))
    observer.boundary("h2d", "start", 0)
    hero_d, board_d = observer.copy_inputs()
    observer.boundary("h2d", "end", 0)
    assert hero_d.set_calls == [((2, 9), cp.stream)]
    assert board_d.set_calls == [((0,), cp.stream)]

    for ordinal in range(2):
        final = DeviceArray(192, "uint8")
        for stage in ("simulate", "reduction"):
            observer.boundary(stage, "start", ordinal)
            observer.boundary(stage, "end", ordinal)
        observer.boundary("d2h", "start", ordinal)
        raw = observer.copy_to_host(final, ordinal)
        observer.boundary("d2h", "end", ordinal)
        assert len(raw) == 192
        assert final.get_calls[0][1:] == (cp.stream, False)

    profile = observer.finish()
    assert profile == {
        "d2h_gpu_ns": "2000000",
        "h2d_gpu_ns": "1000000",
        "reduction_gpu_ns": "2000000",
        "simulate_gpu_ns": "2000000",
    }
    assert cp.events[-1].synchronized is True
