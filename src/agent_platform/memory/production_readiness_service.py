from agent_platform.memory.production_readiness import (
    Layer3ReadinessCheck,
    Layer3ReadinessReport,
    ReadinessStatus,
)


class Layer3ProductionReadinessService:
    """Builds a formal Layer 3 production-readiness report."""

    def evaluate(
        self,
        *,
        tenant_isolation_enforced: bool,
        retrieval_authorization_enforced: bool,
        memory_write_policy_enforced: bool,
        memory_read_policy_enforced: bool,
        citations_validated: bool,
        context_budget_enforced: bool,
        sensitive_payload_logging_disabled: bool,
        retrieval_evaluation_available: bool,
        retrieval_observability_available: bool,
        provider_boundaries_abstracted: bool,
    ) -> Layer3ReadinessReport:
        checks = (
            self._check(
                check_id="tenant_isolation",
                description=("Tenant isolation is enforced."),
                condition=tenant_isolation_enforced,
            ),
            self._check(
                check_id="retrieval_authorization",
                description=(
                    "Retrieval authorization occurs before evidence reaches the model."
                ),
                condition=(retrieval_authorization_enforced),
            ),
            self._check(
                check_id="memory_write_policy",
                description=(
                    "Long-term memory writes are governed by platform policy."
                ),
                condition=memory_write_policy_enforced,
            ),
            self._check(
                check_id="memory_read_policy",
                description=("Memory reads are governed before exposure to the model."),
                condition=memory_read_policy_enforced,
            ),
            self._check(
                check_id="citation_validation",
                description=(
                    "Model-produced citations are validated against trusted evidence."
                ),
                condition=citations_validated,
            ),
            self._check(
                check_id="context_budget",
                description=(
                    "Context admission is bounded by an explicit token budget."
                ),
                condition=context_budget_enforced,
            ),
            self._check(
                check_id="safe_observability",
                description=(
                    "Sensitive prompts and retrieved "
                    "content are excluded from telemetry."
                ),
                condition=(sensitive_payload_logging_disabled),
            ),
            self._check(
                check_id="retrieval_evaluation",
                description=(
                    "Retrieval quality can be measured independently from generation."
                ),
                condition=retrieval_evaluation_available,
            ),
            self._check(
                check_id="retrieval_observability",
                description=(
                    "Memory and retrieval operations have safe operational telemetry."
                ),
                condition=(retrieval_observability_available),
            ),
            self._check(
                check_id="provider_abstraction",
                description=(
                    "Embedding, vector-store and agent "
                    "boundaries remain provider-neutral."
                ),
                condition=provider_boundaries_abstracted,
            ),
        )

        return Layer3ReadinessReport(
            checks=checks,
        )

    @staticmethod
    def _check(
        *,
        check_id: str,
        description: str,
        condition: bool,
    ) -> Layer3ReadinessCheck:
        return Layer3ReadinessCheck(
            check_id=check_id,
            description=description,
            status=(ReadinessStatus.PASS if condition else ReadinessStatus.FAIL),
        )
