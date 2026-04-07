"""Vatavaran graders for hackathon validation.

Each grader corresponds to one task in openenv.yaml and has its scoring
criteria hardcoded. The validator imports these directly, calls grade()
with a prediction string, and verifies:
  - The class is importable
  - grade() returns a float in [0.01, 0.99]
  - Results are deterministic across repeated calls
"""

from __future__ import annotations

from vatavaran.server.evaluator import evaluate_prediction


class _BaseGrader:
    """Shared grading logic. Subclasses set scoring_points for their task."""

    scoring_points: str = ""

    def grade(self, prediction: str) -> float:
        result = evaluate_prediction(prediction or "", self.scoring_points)
        return max(0.01, min(0.99, float(result["score"])))


# ---------------------------------------------------------------------------
# Easy tasks — component identification only
# ---------------------------------------------------------------------------

class EasyGrader1(_BaseGrader):
    scoring_points = "The only predicted root cause component is Tomcat01"


class EasyGrader2(_BaseGrader):
    scoring_points = "The only predicted root cause component is MG02"


class EasyGrader3(_BaseGrader):
    scoring_points = "The only predicted root cause component is Mysql01"


# ---------------------------------------------------------------------------
# Middle tasks — component + reason
# ---------------------------------------------------------------------------

class MiddleGrader1(_BaseGrader):
    scoring_points = (
        "The only predicted root cause component is Tomcat01\n"
        "The only predicted root cause reason is high CPU usage"
    )


class MiddleGrader2(_BaseGrader):
    scoring_points = (
        "The only predicted root cause component is MG02\n"
        "The only predicted root cause reason is network latency"
    )


class MiddleGrader3(_BaseGrader):
    scoring_points = (
        "The only predicted root cause component is Mysql01\n"
        "The only predicted root cause reason is high memory usage"
    )


# ---------------------------------------------------------------------------
# Hard tasks — datetime + component + reason
# ---------------------------------------------------------------------------

class HardGrader1(_BaseGrader):
    scoring_points = (
        "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-05 10:12:00\n"
        "The only predicted root cause component is Tomcat01\n"
        "The only predicted root cause reason is high CPU usage"
    )


class HardGrader2(_BaseGrader):
    scoring_points = (
        "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-06 14:20:00\n"
        "The only predicted root cause component is MG02\n"
        "The only predicted root cause reason is network latency"
    )


class HardGrader3(_BaseGrader):
    scoring_points = (
        "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-07 16:05:00\n"
        "The only predicted root cause component is Mysql01\n"
        "The only predicted root cause reason is high memory usage"
    )
