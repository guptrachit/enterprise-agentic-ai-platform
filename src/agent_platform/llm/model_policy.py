from dataclasses import dataclass

from agent_platform.llm.errors import LLMModelPolicyNotFoundError
from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class ModelPolicy:
    """Maps logical workloads to ordered logical model candidates."""

    assignments: dict[LLMWorkload, str | tuple[str, ...]]

    def models_for(
        self,
        workload: LLMWorkload,
    ) -> tuple[str, ...]:
        """Return ordered logical model candidates for a workload."""

        try:
            assignment = self.assignments[workload]
        except KeyError as error:
            raise LLMModelPolicyNotFoundError(workload.value) from error

        if isinstance(assignment, str):
            return (assignment,)

        return assignment

    def model_for(
        self,
        workload: LLMWorkload,
    ) -> str:
        """Return the primary logical model for a workload."""

        return self.models_for(workload)[0]
