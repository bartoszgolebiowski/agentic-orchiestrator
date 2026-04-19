from __future__ import annotations

import pytest

from engine.core.pipeline import PipelineState, StageStatus, is_failure_observation


class TestPipelineState:
    def test_empty_pipeline_is_complete(self):
        pipeline = PipelineState([])
        assert pipeline.is_empty
        assert pipeline.is_complete
        assert pipeline.next_stage is None
        assert pipeline.missing_stages == []

    def test_initial_state(self):
        pipeline = PipelineState(["a", "b", "c"])
        assert not pipeline.is_empty
        assert not pipeline.is_complete
        assert pipeline.next_stage == "a"
        assert pipeline.missing_stages == ["a", "b", "c"]
        assert pipeline.completed_stages == []

    def test_mark_completed(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.mark_completed("a", "result-a")
        assert pipeline.is_stage_completed("a")
        assert not pipeline.is_complete
        assert pipeline.next_stage == "b"
        assert pipeline.missing_stages == ["b"]
        assert pipeline.completed_stages == ["a"]

    def test_full_completion(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.mark_completed("a", "result-a")
        pipeline.mark_completed("b", "result-b")
        assert pipeline.is_complete
        assert pipeline.next_stage is None
        assert pipeline.missing_stages == []

    def test_mark_failed_does_not_complete(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.mark_failed("a", "some error")
        assert not pipeline.is_stage_completed("a")
        assert pipeline.next_stage == "a"

    def test_observe_result_success(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.observe_result("a", "The extraction returned 42 items.")
        assert pipeline.is_stage_completed("a")

    def test_observe_result_failure(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.observe_result("a", "ERROR: ValueError: something broke")
        assert not pipeline.is_stage_completed("a")
        result = pipeline.get_result("a")
        assert result is not None
        assert result.status == StageStatus.FAILED

    def test_observe_result_failure_json_envelope(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.observe_result("a", '{"status": "error", "hint": "something broke"}')
        assert not pipeline.is_stage_completed("a")
        result = pipeline.get_result("a")
        assert result is not None
        assert result.status == StageStatus.FAILED

    @pytest.mark.parametrize("observation", ["", "   ", "\n\t"])
    def test_observe_result_rejects_blank_observation(self, observation):
        pipeline = PipelineState(["a"])
        pipeline.observe_result("a", observation)
        result = pipeline.get_result("a")
        assert result is not None
        assert result.status == StageStatus.FAILED

    def test_observe_result_ignores_non_pipeline_actions(self):
        pipeline = PipelineState(["a"])
        pipeline.observe_result("unknown_action", "some result")
        assert pipeline.get_result("unknown_action") is None

    def test_observe_result_does_not_overwrite_completed(self):
        pipeline = PipelineState(["a"])
        pipeline.mark_completed("a", "original")
        pipeline.observe_result("a", "ERROR: should not change")
        assert pipeline.is_stage_completed("a")

    def test_summary(self):
        pipeline = PipelineState(["a", "b"])
        pipeline.mark_completed("a", "ok")
        summary = pipeline.summary()
        assert summary["stages"] == ["a", "b"]
        assert summary["completed"] == ["a"]
        assert summary["missing"] == ["b"]
        assert summary["is_complete"] is False


class TestIsFailureObservation:
    @pytest.mark.parametrize("text", [
        "ERROR: ValueError: boom",
        "error: something",
        "I'm unable to process this",
        "I am unable to do it",
        "I cannot complete the request",
        "Could not find the file",
        "Failed to execute tool",
        '{"status": "error", "hint": "tool failed"}',
    ])
    def test_detects_failures(self, text):
        assert is_failure_observation(text) is True

    @pytest.mark.parametrize("text", [
        "The extraction returned 42 items.",
        "Successfully completed the task.",
        '{"result": "ok"}',
        "Here is the analysis: no errors found",
    ])
    def test_passes_successes(self, text):
        assert is_failure_observation(text) is False
