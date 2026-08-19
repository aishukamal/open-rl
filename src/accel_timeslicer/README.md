# Accelerator Time-Slicing

OpenRL no longer ships its own time-slicing coordinator. Full fine-tuning
workers that share an accelerator delegate coordination to the llm-d
time-slicing platform ("llmd-app" mode, the default); the internal
accel-timeslicer daemon, its socket protocol, and its checkpoint/restore
backends have been removed.

This package now contains:

- `workload.py` — workload identity (`WorkloadRef`, job id / group naming)
  shared by the worker launchers and the coordination clients.
- `time_slicer.py` — the worker-facing `TimeSlicerClient` protocol,
  `time_slicer_client_from_env()` (mode dispatch), and the no-op client for
  mode `off`.
- `llmd_app.py` — the llm-d platform integration (see below).

## llmd-app mode (default)

`time_slicer_client_from_env()` returns an adapter over the cluster-scoped
llm-d TimeSlice Orchestrator (`OPEN_RL_TIME_SLICE_ORCH_ADDR`), and workers
register their suspend/resume mechanics with the node-local llm-d Snapshot
Agent (`OPEN_RL_SNAPSHOT_AGENT_ADDR`) over the `app_channel` stream — trainers
hand over their pinned-buffer offload/reload callbacks, samplers hand over the
vLLM engine object. The orchestrator defers the snapshot until another job
acquires the group lock (zero-overhead when nobody is waiting), and its queue
is FIFO.

Identity: the same job_id/group pair is used for the orchestrator lock, the
app_channel registration, and the `timeslice.io/job-id` / `timeslice.io/group`
pod labels the Snapshot Agent's Kubernetes watcher uses to track job state.

## off mode

`OPEN_RL_TIME_SLICE_MODE=off` disables coordination for single-tenant
deployments that own their GPU outright: acquire/release become no-ops and
the worker loops manage their own offload (inline `sleep()`/`wake_up()`
around each GPU work unit).
