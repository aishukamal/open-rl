import os
import unittest
from unittest.mock import patch

from accel_timeslicer.llmd_app import is_llmd_app_mode, timeslice_mode
from accel_timeslicer.time_slicer import NullTimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import DEFAULT_TIME_SLICE_GROUP, WorkloadRef


class TimeSliceModeTest(unittest.TestCase):
  def test_default_mode_is_llmd_app(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(timeslice_mode(), "llmd-app")
      self.assertTrue(is_llmd_app_mode())

  def test_off_mode_returns_null_client(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICE_MODE": "off"}):
      client = time_slicer_client_from_env()
      self.assertIsInstance(client, NullTimeSlicerClient)

  def test_unknown_mode_raises_with_guidance(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICE_MODE": "legacy"}), self.assertRaisesRegex(RuntimeError, "llmd-app"):
      time_slicer_client_from_env()


class NullClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_acquire_yields_none_and_register_is_noop(self) -> None:
    client = NullTimeSlicerClient()
    workload = WorkloadRef(job_id="job-a", group="trainers")
    self.assertEqual((await client.register(workload)).get("ok"), True)
    async with client.acquire(workload) as result:
      self.assertIsNone(result)
    self.assertEqual((await client.unregister(workload)).get("ok"), True)
    await client.close()


class WorkloadFromEnvTest(unittest.TestCase):
  def test_env_job_id_wins(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICE_JOB_ID": "env-job"}):
      workload = workload_from_env(101, job_id="model-a")
      self.assertEqual(workload.job_id, "env-job")

  def test_explicit_job_id_and_default_group(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      workload = workload_from_env(101, job_id="model-a")
      self.assertEqual(workload.job_id, "model-a")
      self.assertEqual(workload.group, DEFAULT_TIME_SLICE_GROUP)

  def test_pid_fallback(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      workload = workload_from_env(101)
      self.assertEqual(workload.job_id, "101")


if __name__ == "__main__":
  unittest.main()
