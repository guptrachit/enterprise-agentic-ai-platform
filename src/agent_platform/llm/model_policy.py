from dataclasses import dataclass

from agent_platform.llm.errors import LLMModelPolicyNotFoundError
from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class ModelPolicy:
    """Maps logical workloads to logical model names."""

    assignments: dict[LLMWorkload, str]

    def model_for(
        self,
        workload: LLMWorkload,
    ) -> str:
        """Return the logical model name assigned to a workload."""

        try:
            return self.assignments[workload]
        except KeyError as error:
            raise LLMModelPolicyNotFoundError(workload.value) from error
