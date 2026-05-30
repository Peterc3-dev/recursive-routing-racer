"""Execution belts — CPU ThreadPoolExecutor + Vulkan Kompute GPU belt + assembly queue.

Implements the assembly line model: CPU and GPU produce fragments asynchronously,
assembly station combines them.
"""

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .dispatcher import Device, Dispatcher
from .pulse import PulseController
from .monitor import HardwareMonitor
from .npu_belt import NpuExecutionBelt


@dataclass
class WorkItem:
    """A unit of work to be executed on a device."""
    operation: str
    fn: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    input_size: int = 0
    priority: int = 0  # lower = higher priority


@dataclass
class Fragment:
    """Result fragment from a belt execution."""
    device: Device
    operation: str
    result: Any = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = 0.0


class CpuBelt:
    """CPU execution belt using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = None):
        import os
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or min(4, os.cpu_count() or 2)
        )
        self._active = 0
        self._lock = threading.Lock()

    def submit(self, work: WorkItem) -> Future:
        with self._lock:
            self._active += 1

        def _run():
            start = time.perf_counter()
            try:
                result = work.fn(*work.args, **work.kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                return Fragment(
                    device=Device.CPU,
                    operation=work.operation,
                    result=result,
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                return Fragment(
                    device=Device.CPU,
                    operation=work.operation,
                    error=str(e),
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
            finally:
                with self._lock:
                    self._active -= 1

        return self._pool.submit(_run)

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active

    def shutdown(self):
        self._pool.shutdown(wait=False)


class GpuBelt:
    """GPU execution belt — runs work items through Vulkan/Kompute.

    Work items are functions that accept a Kompute manager and return results.
    Execution is serialized (GPU is single-context) but wrapped for async interface.
    """

    def __init__(self, pulse: PulseController, monitor: HardwareMonitor):
        self._pulse = pulse
        self._monitor = monitor
        self._queue: queue.Queue[tuple[WorkItem, Future]] = queue.Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._running = False
        self._kp_mgr = None

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)  # sentinel

    def submit(self, work: WorkItem) -> Future:
        future: Future = Future()
        self._queue.put((work, future))
        return future

    def _run_loop(self):
        # Lazy init Kompute
        self._init_kompute()

        while self._running:
            item = self._queue.get()
            if item is None:
                break

            work, future = item
            snap = self._monitor.snapshot
            temp = snap.gpu.temp_c

            # Check pulse controller
            if not self._pulse.should_fire_gpu(temp):
                cooldown = self._pulse.effective_cooldown_ms(temp) / 1000.0
                time.sleep(cooldown)
                self._pulse.record_cooldown(cooldown * 1000)

            start = time.perf_counter()
            try:
                result = work.fn(*work.args, **work.kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                self._pulse.record_burst(elapsed)
                fragment = Fragment(
                    device=Device.GPU,
                    operation=work.operation,
                    result=result,
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
                future.set_result(fragment)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                fragment = Fragment(
                    device=Device.GPU,
                    operation=work.operation,
                    error=str(e),
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
                future.set_result(fragment)

    def _init_kompute(self):
        try:
            import kp
            self._kp_mgr = kp.Manager()
        except ImportError:
            pass  # Kompute not available — GPU belt will run CPU fallbacks


class NpuBelt:
    """NPU execution belt — runs inference via FastFlowLM (FLM) server.

    FLM exposes an OpenAI-compatible API on localhost. Work items are
    HTTP requests to the FLM server. Falls back to CPU if NPU is unavailable.
    """

    def __init__(self, monitor: HardwareMonitor, flm_port: int = 8899,
                 execution_belt: NpuExecutionBelt = None):
        self._monitor = monitor
        self._port = flm_port
        self._queue: queue.Queue[tuple[WorkItem, Future]] = queue.Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._running = False
        self._flm_available = False
        self.execution_belt = execution_belt or NpuExecutionBelt()

    def start(self):
        self._running = True
        self._flm_available = self._check_flm()
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)

    @property
    def available(self) -> bool:
        return self._flm_available or self.execution_belt.npu_available

    @property
    def belt_status(self) -> dict:
        """Status of the NPU execution belt backends."""
        return self.execution_belt.status()

    def submit(self, work: WorkItem) -> Future:
        future: Future = Future()
        self._queue.put((work, future))
        return future

    def _run_loop(self):
        while self._running:
            item = self._queue.get()
            if item is None:
                break

            work, future = item
            start = time.perf_counter()
            try:
                result = work.fn(*work.args, **work.kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                fragment = Fragment(
                    device=Device.NPU,
                    operation=work.operation,
                    result=result,
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
                future.set_result(fragment)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                fragment = Fragment(
                    device=Device.NPU,
                    operation=work.operation,
                    error=str(e),
                    duration_ms=elapsed,
                    timestamp=time.time(),
                )
                future.set_result(fragment)

    def _check_flm(self) -> bool:
        """Check if NPU is available (driver bound + flm binary exists)."""
        # Direct check — don't rely on monitor which may not have polled yet
        from pathlib import Path
        if not Path("/dev/accel/accel0").exists():
            return False
        import shutil
        return shutil.which("flm") is not None


class AssemblyStation:
    """Collects fragments from CPU and GPU belts, combines into output."""

    def __init__(self):
        self._fragments: list[Fragment] = []
        self._lock = threading.Lock()

    def collect(self, fragment: Fragment):
        with self._lock:
            self._fragments.append(fragment)

    def assemble(self, expected: int, timeout_s: float = 30.0) -> list[Fragment]:
        """Wait for expected number of fragments, then return them."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if len(self._fragments) >= expected:
                    result = self._fragments[:expected]
                    self._fragments = self._fragments[expected:]
                    return result
            if time.monotonic() > deadline:
                with self._lock:
                    result = list(self._fragments)
                    self._fragments.clear()
                    return result
            time.sleep(0.001)

    def clear(self):
        with self._lock:
            self._fragments.clear()


class Executor:
    """Main executor — dispatches work to CPU/GPU/NPU belts, assembles results."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        pulse: PulseController,
        monitor: HardwareMonitor,
        cpu_workers: int = None,
    ):
        self.dispatcher = dispatcher
        self.cpu_belt = CpuBelt(max_workers=cpu_workers)
        self.gpu_belt = GpuBelt(pulse=pulse, monitor=monitor)
        self._npu_execution_belt = NpuExecutionBelt(personality=None)
        self.npu_belt = NpuBelt(monitor=monitor, execution_belt=self._npu_execution_belt)
        self.assembly = AssemblyStation()
        self._monitor = monitor

    def start(self):
        self.gpu_belt.start()
        self.npu_belt.start()
        # Update dispatcher with NPU availability
        self.dispatcher.npu_available = self.npu_belt.available

    def stop(self):
        self.gpu_belt.stop()
        self.npu_belt.stop()
        self.cpu_belt.shutdown()

    def execute(self, work: WorkItem) -> Fragment:
        """Execute a single work item on the best available device."""
        decision = self.dispatcher.dispatch(work.operation, work.input_size)

        if decision.device == Device.NPU and self.npu_belt.available:
            future = self.npu_belt.submit(work)
        elif decision.device == Device.GPU:
            future = self.gpu_belt.submit(work)
        else:
            future = self.cpu_belt.submit(work)

        fragment = future.result(timeout=60.0)

        # Record for learning
        self.dispatcher.record_result(
            device=fragment.device,
            operation=fragment.operation,
            duration_ms=fragment.duration_ms,
            input_size=work.input_size,
            success=fragment.error is None,
        )

        return fragment

    def execute_parallel(self, works: list[WorkItem]) -> list[Fragment]:
        """Execute multiple work items, dispatching each to the best device."""
        futures = []
        for work in works:
            decision = self.dispatcher.dispatch(work.operation, work.input_size)
            if decision.device == Device.NPU and self.npu_belt.available:
                futures.append((work, self.npu_belt.submit(work)))
            elif decision.device == Device.GPU:
                futures.append((work, self.gpu_belt.submit(work)))
            else:
                futures.append((work, self.cpu_belt.submit(work)))

        fragments = []
        for work, future in futures:
            fragment = future.result(timeout=60.0)
            self.dispatcher.record_result(
                device=fragment.device,
                operation=fragment.operation,
                duration_ms=fragment.duration_ms,
                input_size=work.input_size,
                success=fragment.error is None,
            )
            fragments.append(fragment)

        return fragments
