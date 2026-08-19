import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from .workload import DEFAULT_TIME_SLICE_GROUP, WorkloadRef


class TimeSlicerClient(Protocol):
  async def register(self, workload: WorkloadRef) -> dict[str, Any]: ...

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]: ...

  def acquire(self, workload: WorkloadRef) -> AbstractAsyncContextManager[Any]: ...

  async def close(self) -> None: ...


class NullTimeSlicerClient:
  """No-op client for OPEN_RL_TIME_SLICE_MODE=off.

  Single-tenant deployments that own their GPU outright need no cross-job
  coordination: acquire() yields immediately (with None, so callers can tell
  no coordinator is present) and the worker loops fall back to managing their
  own offload — inline sleep()/wake_up() around each GPU work unit.
  """

  async def register(self, workload: WorkloadRef) -> dict[str, Any]:
    return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, workload: WorkloadRef) -> AsyncIterator[None]:
    yield None

  async def close(self) -> None:
    return None


def workload_from_env(pid: int | None = None, job_id: str | None = None, group: str = DEFAULT_TIME_SLICE_GROUP) -> WorkloadRef:
  env_job_id = os.getenv("OPEN_RL_TIME_SLICE_JOB_ID")
  if env_job_id:
    return WorkloadRef(job_id=env_job_id, group=group)
  if job_id:
    return WorkloadRef(job_id=job_id, group=group)
  if pid is None:
    raise ValueError("workload requires job_id")
  return WorkloadRef(job_id=str(pid), group=group)


def time_slicer_client_from_env() -> TimeSlicerClient:
  from .llmd_app import is_llmd_app_mode, timeslice_mode

  if is_llmd_app_mode():
    from .llmd_app import OrchestratorTimeSlicerClient

    return OrchestratorTimeSlicerClient()

  mode = timeslice_mode()
  if mode in ("off", "none"):
    return NullTimeSlicerClient()

  raise RuntimeError(
    f"Unknown OPEN_RL_TIME_SLICE_MODE '{mode}'. The internal accel-timeslicer "
    "daemon has been removed; use 'llmd-app' (default, coordination via the "
    "llm-d time-slicing platform) or 'off' (single-tenant, self-managed offload)."
  )
