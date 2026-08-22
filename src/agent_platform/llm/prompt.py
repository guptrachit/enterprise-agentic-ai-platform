from dataclasses import dataclass
from string import Formatter

from agent_platform.llm.errors import LLMPromptVariableError


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned template used to construct an LLM prompt."""

    name: str
    version: str
    template: str

    @property
    def required_variables(self) -> frozenset[str]:
        """Return variables required by the template."""

        variables: set[str] = set()

        for _, field_name, _, _ in Formatter().parse(self.template):
            if field_name:
                variables.add(field_name)

        return frozenset(variables)

    def render(self, **variables: object) -> str:
        """Render the template using supplied variables."""

        for variable_name in self.required_variables:
            if variable_name not in variables:
                raise LLMPromptVariableError(variable_name)

        return self.template.format(**variables)
