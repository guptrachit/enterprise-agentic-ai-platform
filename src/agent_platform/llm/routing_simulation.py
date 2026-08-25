from dataclasses import dataclass

from agent_platform.llm.model_capability import ModelCapability
from agent_platform.llm.model_definition import ModelDefinition
from agent_platform.llm.model_preference import ModelPreference
from agent_platform.llm.routing_constraints import RoutingConstraints
from agent_platform.llm.routing_decision import RoutingDecision
from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class RoutingSimulationCase:
    """One offline model-routing simulation input."""

    name: str
    workload: LLMWorkload
    constraints: RoutingConstraints | None = None
    required_capabilities: frozenset[ModelCapability] = frozenset()
    preference: ModelPreference | None = None
    expected_selected_model: str | None = None


@dataclass(frozen=True)
class RoutingSimulationResult:
    """Result of simulating one routing case."""

    case: RoutingSimulationCase
    decision: RoutingDecision

    @property
    def selected_model(self) -> ModelDefinition:
        """Return the model selected by the simulated policy."""

        return self.decision.selected_model

    @property
    def matches_expectation(self) -> bool | None:
        """Return whether actual selection matches the declared expectation."""

        if self.case.expected_selected_model is None:
            return None

        return self.selected_model.name == self.case.expected_selected_model
