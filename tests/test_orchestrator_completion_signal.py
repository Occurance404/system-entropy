from src.orchestrator.core.orchestrator import Orchestrator


def _detect(text):
    orchestrator = object.__new__(Orchestrator)
    return orchestrator._agent_signaled_completion(text)


def test_detects_explicit_task_complete_markers():
    assert _detect("Task is complete. Output saved.")
    assert _detect("TASK COMPLETE")
    assert _detect("task_complete: true")
    assert _detect("Mission accomplished.")


def test_ignores_loose_progress_phrases():
    assert not _detect("Here is a final summary of progress so far.")
    assert not _detect("I completed the analysis section, next I will edit files.")
    assert not _detect("Task completion is pending further validation.")
