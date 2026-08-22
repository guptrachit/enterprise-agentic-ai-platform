from dataclasses import dataclass

from agent_platform.llm.workload import LLMWorkload


@dataclass(frozen=True)
class ModelDefinition:
    """Platform definition of an available LLM model."""

    name: str
    provider: str
    provider_model: str
    workloads: frozenset[LLMWorkload]
    supports_structured_output: bool = True
    enabled: bool = True

    def supports_workload(
        self,
        workload: LLMWorkload,
    ) -> bool:
        """Return whether this model supports the requested workload."""

        return workload in self.workloads
