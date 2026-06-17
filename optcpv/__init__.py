"""OptCPV: a CV-native schematic drawing optimizer."""

from .artifact import draw_artifact, draw_svg
from .models import Circuit, Component, LayoutSupport, SchematicArtifact
from .optimizer import draw_optimized_artifact, draw_optimized_svg
from .planning_agent import FakePlanningClient, GeminiPlanningClient, SemanticPlanningClient
from .planning_hints import (
    GridPlacementHint,
    LaneHint,
    PlanningHints,
    RoutePolicyHint,
    SchematicLayoutHints,
    SchematicPlanningRequest,
    StageHint,
)
from .planner import plan_layout
from .semantic_repair import repair_circuit
from .visual_review import (
    FakeVisualReviewClient,
    GeminiVisualReviewClient,
    HeuristicVisualReviewClient,
    VisualError,
    VisualPatch,
    VisualReview,
    VisualReviewClient,
)
from .vision_agent import GeminiVisionClient, HeuristicVisionClient, VisualFeedback, heuristic_feedback

__all__ = [
    "Circuit",
    "Component",
    "FakePlanningClient",
    "FakeVisualReviewClient",
    "GeminiVisualReviewClient",
    "GeminiVisionClient",
    "GeminiPlanningClient",
    "GridPlacementHint",
    "HeuristicVisionClient",
    "HeuristicVisualReviewClient",
    "LaneHint",
    "LayoutSupport",
    "PlanningHints",
    "RoutePolicyHint",
    "SchematicArtifact",
    "SchematicLayoutHints",
    "SchematicPlanningRequest",
    "SemanticPlanningClient",
    "StageHint",
    "VisualFeedback",
    "VisualError",
    "VisualPatch",
    "VisualReview",
    "VisualReviewClient",
    "draw_artifact",
    "draw_optimized_artifact",
    "draw_optimized_svg",
    "draw_svg",
    "heuristic_feedback",
    "plan_layout",
    "repair_circuit",
]
