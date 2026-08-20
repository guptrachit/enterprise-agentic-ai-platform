# Layer 1 — LLM Foundation
## Progress, Architecture Decisions, Technical Decisions & Interview Preparation

> Permanent engineering record for Layer 1 of the Enterprise Agentic AI Platform.

---

# 1. Layer 1 Objective

Layer 1 establishes the production engineering foundation for communicating
with Large Language Models (LLMs).

The objective is not simply to call an LLM API. The objective is to build a
provider-independent, production-oriented LLM integration layer.

The foundation addresses:

- LLM abstraction
- Provider integration
- Configuration
- Token usage tracking
- Cost awareness
- Error normalization
- Retry behavior
- Exponential backoff
- Jitter
- Retry budgets
- Request timeouts
- Provider-specific retry guidance
- Observability metadata
- Automated testing

---

# 2. Architecture Implemented

The current Layer 1 architecture is:

    Application
         |
         v
    LLMClient abstraction
         |
         v
    OpenAIClient
         |
         +------------------+
         |                  |
         v                  v
    Error Mapper       Retry Executor
                            |
                 +----------+----------+
                 |          |          |
              Timeout    Backoff    Budget
                            |
                          Jitter
                            |
                            v
                    OpenAI Responses API
                            |
                            v
                       LLMResponse

The application therefore does not directly depend on OpenAI-specific
exceptions or retry mechanics.

---

# 3. Major Components Implemented

## 3.1 LLM Abstraction

The platform defines an LLM client abstraction instead of allowing
application code to directly call OpenAI.

### Why

This provides:

- Provider independence
- Easier testing
- Easier future provider integration
- Separation of business logic from infrastructure
- Cleaner architecture

### Interview Question

How would you design an application that can switch between OpenAI,
Anthropic, Azure OpenAI, or another model provider?

### Answer

Use an interface or abstract LLM client and place provider-specific
implementations behind that interface.

---

# 4. OpenAI Client

Implemented in:

`src/agent_platform/llm/openai_client.py`

The OpenAI client is responsible for:

- Creating the asynchronous OpenAI client
- Selecting the model
- Sending prompts
- Measuring latency
- Extracting token usage
- Returning a normalized response
- Mapping provider errors
- Executing requests through the retry framework

Current model:

`gpt-5-mini`

The OpenAI Responses API is used.

---

# 5. Why Async Was Chosen

The client uses:

`AsyncOpenAI`

LLM API calls are primarily network I/O operations.

Async execution allows the application to perform other work while waiting
for the model provider.

This becomes increasingly important for:

- Concurrent agent execution
- Multiple tool calls
- Parallel model calls
- Streaming
- Agent orchestration
- Higher request volumes

### Interview Question

Why use async for LLM applications?

### Answer

LLM calls are network-bound operations with potentially significant latency.
Async I/O allows better resource utilization and concurrency without blocking
worker threads.

---

# 6. Normalized LLM Response

The provider-specific OpenAI response is converted into the platform's
normalized `LLMResponse`.

The response contains:

- Generated text
- Input token count
- Output token count
- Total token count
- Provider
- Model
- Latency
- Request ID

### Why

Provider-specific response structures should not leak into the rest of the
application.

This creates a stable platform-level contract.

---

# 7. Token Usage Tracking

The platform captures:

- `input_tokens`
- `output_tokens`
- `total_tokens`

### Why

Token usage is required for:

- Cost management
- Usage analytics
- Model comparison
- Capacity planning
- FinOps
- Observability

### Interview Question

How would you control LLM costs in an enterprise AI platform?

### Answer

Track token usage at request level and associate usage with model, provider,
and eventually tenant, application, user, agent, or business workflow.

---

# 8. Latency Tracking

Request latency is measured using:

`time.perf_counter()`

`perf_counter()` is appropriate for elapsed-time measurement.

Latency is important for:

- SLA monitoring
- Model comparison
- Performance optimization
- Agent latency analysis
- User experience monitoring

---

# 9. Request ID Tracking

The provider response ID is captured as request metadata.

### Purpose

- Debugging
- Correlation
- Provider-side investigation
- Observability
- Incident troubleshooting

A future platform-wide correlation ID can be added on top of this.

---

# 10. Configuration

The platform uses a centralized Settings object.

The OpenAI API key is supplied through configuration rather than hardcoded
into the client.

### Why

Secrets should not be stored in source code.

This supports:

- Local development
- CI/CD
- Production environments
- Secret managers
- Environment-specific configuration

---

# 11. Error Abstraction

Provider-specific errors are translated into platform-level errors.

Examples include:

- `LLMTimeoutError`
- `LLMRateLimitError`
- `LLMInvalidRequestError`
- `LLMTransientError`
- `LLMConfigurationError`
- `LLMRetryBudgetExceededError`

### Why

Application code should not need to understand OpenAI SDK exception types.

The architecture becomes:

    OpenAI Exception
           |
           v
      Error Mapper
           |
           v
    Platform LLM Error
           |
           v
    Retry / Application Logic

This is an important infrastructure boundary.

---

# 12. Error Mapping

Implemented in:

`src/agent_platform/llm/error_mapper.py`

The mapper translates OpenAI exceptions such as:

- Authentication errors
- Rate limit errors
- Bad requests
- Timeout errors
- Connection errors
- Internal server errors

into platform-level errors.

---

# 13. Retry Classification

Not every error should be retried.

### Retryable

Examples:

- Rate limiting
- Temporary provider failures
- Connection failures

### Non-retryable

Examples:

- Invalid request
- Authentication/configuration failure

### Why

Blind retries can increase:

- Latency
- Cost
- Provider load
- Failure duration

The retry framework therefore depends on error classification.

---

# 14. Retry Policy

Implemented in:

`src/agent_platform/llm/retry.py`

The policy controls:

- Maximum attempts
- Initial backoff
- Maximum backoff
- Retry budget
- Attempt timeout
- Jitter

Current defaults:

- `max_attempts = 3`
- `initial_backoff_seconds = 0.5`
- `max_backoff_seconds = 5.0`
- `max_retry_budget_seconds = 10.0`
- `attempt_timeout_seconds = 30.0`
- `jitter = True`

These are engineering defaults and can be tuned later using production
telemetry.

---

# 15. Exponential Backoff

Retry delay follows an exponential pattern.

Conceptually:

    Retry 1 -> 0.5 seconds
    Retry 2 -> 1.0 seconds
    Retry 3 -> 2.0 seconds

The delay is capped by:

`max_backoff_seconds`

### Why

Immediate repeated retries can create a retry storm.

Exponential backoff gives the provider time to recover.

### Interview Question

What is exponential backoff and why is it used?

### Answer

It progressively increases the wait between retry attempts to reduce load
during transient failures and avoid continuously hammering a failing service.

---

# 16. Jitter

Random jitter is supported.

Instead of every client retrying at exactly the same time, jitter
randomizes the delay.

### Why

Without jitter, many clients experiencing the same failure can retry
simultaneously.

This can create a:

"thundering herd"

or:

"retry storm"

Jitter distributes retries over time.

---

# 17. Provider Retry-After

The error mapper extracts:

`Retry-After`

from provider HTTP responses when available.

If the provider explicitly tells us when to retry, that value takes precedence
over locally calculated exponential backoff.

### Architecture Principle

Provider guidance takes precedence over local retry calculation.

This is an important production reliability decision.

---

# 18. Retry Budget

The retry executor enforces:

`max_retry_budget_seconds`

The system does not retry indefinitely even when attempts remain.

### Why

Attempt count alone does not protect application latency.

For example:

    Maximum attempts = 5
    Retry budget = 10 seconds

If the next delay would exceed the remaining budget, retrying stops.

This protects the application from excessive retry latency.

---

# 19. Attempt Timeout

Each individual LLM attempt has:

`attempt_timeout_seconds`

The request is wrapped using:

`asyncio.wait_for(...)`

### Why

A network call should not be allowed to hang indefinitely.

This protects:

- Worker resources
- Application responsiveness
- SLA
- Retry behavior

### Important distinction

Attempt timeout controls one request.

Retry budget controls the overall retry process.

---

# 20. Retry Executor

Implemented in:

`src/agent_platform/llm/retry_executor.py`

Responsibilities:

1. Execute the operation
2. Apply per-attempt timeout
3. Detect retryable LLM errors
4. Stop immediately for non-retryable errors
5. Respect maximum attempts
6. Calculate retry delay
7. Respect provider Retry-After
8. Respect total retry budget
9. Sleep before retry
10. Return the successful result

This creates a reusable reliability component independent of OpenAI.

---

# 21. Separation of Responsibilities

The architecture intentionally separates:

    OpenAI Client
        |
        +--> Provider communication

    Error Mapper
        |
        +--> Provider-specific error translation

    Retry Executor
        |
        +--> Retry orchestration

    Retry Policy
        |
        +--> Retry configuration/calculation

    LLM Errors
        |
        +--> Platform error contract

This follows the Single Responsibility Principle.

---

# 22. Testing Strategy

Automated unit tests were created for the major Layer 1 components.

Current test suite:

`30 tests`

Tests cover:

- Configuration
- Cost calculation
- Error mapping
- Error behavior
- Health endpoint
- OpenAI client
- Retry policy
- Retry executor

OpenAI client tests cover:

- Successful response
- Token usage
- Response metadata
- Transient retry
- Non-retryable errors
- Rate-limit retry
- Platform transient errors

---

# 23. Important Testing Lesson

During development, several test failures occurred.

One important failure occurred when test exceptions were passed through the
OpenAI error mapper.

For example, a generic:

`RuntimeError`

was converted to:

`LLMError`

which is non-retryable.

This demonstrated that the retry executor was correctly following its
contract, while the test setup was bypassing the provider abstraction.

The test design was then aligned with the architecture so that the request
boundary produces platform-level LLM errors before the retry executor sees
them.

This was an important architecture validation.

---

# 24. Quality Gates

The repository uses:

    uv run pytest
    uv run ruff check .
    uv run ruff format --check .

Current status:

- 30 tests passing
- Ruff checks passing
- Formatting validation passing

These commands form the local engineering quality gate.

---

# 25. Technical Components Used

## Python

Primary implementation language.

Why:

- Strong AI/ML ecosystem
- Excellent API ecosystem
- Async support
- Testing ecosystem
- Data and AI libraries

## uv

Used for:

- Environment management
- Dependency installation
- Dependency locking
- Running project commands

## pytest

Used for automated testing.

## pytest-asyncio

Used for asynchronous test execution.

## Ruff

Used for:

- Linting
- Import organization
- Formatting

## FastAPI

Used as the application/API foundation.

## OpenAI SDK

Used for provider integration.

---

# 26. Architecture Decisions Made

## Decision 1 — Provider abstraction

Do not expose OpenAI directly to application code.

Reason:

Allows future providers and easier testing.

Status:

Implemented.

---

## Decision 2 — Async LLM client

Use asynchronous OpenAI API calls.

Reason:

LLM requests are network-bound and future agent workloads will require
concurrency.

Status:

Implemented.

---

## Decision 3 — Centralized error model

Normalize provider errors into platform-level errors.

Reason:

Avoid provider coupling.

Status:

Implemented.

---

## Decision 4 — Centralized retry executor

Retry behavior belongs in a reusable infrastructure component rather than
inside every provider client.

Reason:

Avoid duplicated retry implementations.

Status:

Implemented.

---

## Decision 5 — Retry only transient failures

Do not retry invalid requests or authentication failures.

Reason:

Retries cannot fix deterministic failures.

Status:

Implemented.

---

## Decision 6 — Provider Retry-After takes precedence

Honor provider retry guidance when available.

Reason:

The provider has better knowledge of rate-limit recovery timing.

Status:

Implemented.

---

## Decision 7 — Retry budget

Use both attempt limits and total retry budget.

Reason:

Attempt count alone does not protect latency.

Status:

Implemented.

---

## Decision 8 — Attempt timeout

Each LLM request has a maximum execution time.

Reason:

Prevent indefinite network waits.

Status:

Implemented.

---

# 27. Architecture Decisions Intentionally Deferred

The following decisions were intentionally NOT made yet.

## Streaming

Not implemented.

Reason:

Streaming changes the response contract and requires handling for partial
output, cancellation, metrics, and errors.

---

## Tool Calling

Not implemented.

Reason:

Tool calling belongs to the agent/tool orchestration layer.

---

## Structured Outputs

Not yet implemented as a generalized platform capability.

Future need:

Typed model responses and schema validation.

---

## Prompt Management

Not implemented yet.

Future capabilities may include:

- Prompt templates
- Versioning
- Prompt metadata
- Evaluation
- Experimentation

---

## Model Routing

Not implemented.

Future routing could consider:

- Cost
- Latency
- Capability
- Context length
- Reliability
- Data sensitivity

---

## Multi-provider Routing

Not implemented yet.

The current abstraction prepares the architecture for it.

---

## Circuit Breaker

Not implemented yet.

A future circuit breaker could temporarily stop calls to an unhealthy
provider instead of repeatedly attempting requests.

---

## Distributed Rate Limiting

Not implemented.

Current retry policy is local to the process.

A distributed platform will eventually require centralized rate limiting or
provider quota management.

---

## Persistent Observability

Request metadata is currently captured in the response.

It is not yet persisted into a centralized telemetry system.

Future possibilities include:

- OpenTelemetry
- Metrics
- Logs
- Traces
- Prometheus
- Cloud monitoring

---

## LLM Evaluation

Not implemented yet.

Future layers should introduce:

- Quality evaluation
- Golden datasets
- Regression tests
- Hallucination evaluation
- Safety evaluation
- Agent trajectory evaluation

---

# 28. Interview Questions Addressed

Layer 1 has already prepared answers for many senior AI engineering
interview questions.

## Architecture

- How would you design an enterprise LLM abstraction?
- How do you avoid vendor lock-in?
- How would you support multiple LLM providers?
- Where should retry logic live?
- How do you separate provider infrastructure from application logic?

## Reliability

- How should LLM API failures be handled?
- Which errors should be retried?
- What is exponential backoff?
- Why is jitter necessary?
- What is a retry storm?
- What is a retry budget?
- Why use request timeouts?
- What is the difference between timeout and retry budget?
- How should HTTP Retry-After be handled?

## Performance

- Why use async?
- How do you measure LLM latency?
- How would you scale concurrent LLM requests?

## Cost

- How do you track token usage?
- How would you calculate LLM cost?
- How would you optimize model selection?
- How would you implement AI FinOps?

## Testing

- How do you test external LLM integrations without real API calls?
- How do you test retry behavior?
- How do you test timeout behavior?
- How do you test non-retryable errors?
- How do you test provider-specific failures?

## Production Engineering

- How do you protect API keys?
- How do you implement observability?
- How do you correlate LLM requests?
- How would you introduce circuit breakers?
- How would you implement multi-provider failover?

---

# 29. Current Layer 1 Maturity

The implementation has moved beyond a basic:

"call OpenAI and return text"

wrapper.

It now has the foundation of a production-oriented LLM infrastructure layer:

    Application
         |
         v
    LLM Abstraction
         |
         v
    OpenAI Client
         |
         +-------------------+
         |                   |
         v                   v
    Error Mapping       Retry Executor
                              |
                     +--------+--------+
                     |        |        |
                  Timeout  Backoff   Budget
                              |
                            Jitter
                              |
                              v
                     OpenAI Responses API

---

# 30. What Layer 1 Has Taught

The most important engineering lesson is:

> An enterprise AI platform is not simply an LLM API wrapper.

The platform must provide:

- Abstraction
- Reliability
- Security
- Observability
- Cost awareness
- Testability
- Provider independence

The LLM provider is an infrastructure dependency.

The platform owns the engineering contract around that dependency.

---

# 31. Recommended Next Component

Before introducing agents, RAG, tools, or complex orchestration, the next
recommended component is:

## Structured LLM Output / Response Contract

This should establish how the platform handles:

- Typed responses
- JSON/schema validation
- Model output parsing
- Invalid structured responses
- Validation failures
- Retry versus non-retry behavior

This will provide the foundation needed later for:

- Agents
- Tool calls
- Planning
- Classification
- Routing
- Structured decisions

---

# 32. Definition of Done

Layer 1 foundation currently satisfies:

- [x] LLM abstraction exists
- [x] OpenAI provider integration exists
- [x] Async execution exists
- [x] Token usage is captured
- [x] Latency is captured
- [x] Request ID is captured
- [x] Provider errors are normalized
- [x] Retry policy exists
- [x] Exponential backoff exists
- [x] Jitter exists
- [x] Retry-After support exists
- [x] Retry budget exists
- [x] Attempt timeout exists
- [x] Retry executor exists
- [x] Unit tests exist
- [x] Linting passes
- [x] Formatting passes

Future capabilities remain intentionally deferred.

---

# 33. Engineering Principle

> Build the reliability and abstraction layer before building intelligence
> and orchestration on top of it.

The later agentic AI layers will depend on this foundation.
## Component 2 — LLM Configuration and Provider Factory

### 2.1 Objective

The second Layer 1 component introduced configuration-driven LLM provider selection and separated application-level LLM usage from provider-specific implementations.

The objective was to ensure that application code depends on the platform-level `LLMClient` abstraction rather than directly depending on `OpenAIClient`.

This establishes the foundation for supporting multiple LLM providers without requiring changes to business/application logic.

### 2.2 Problem Addressed

Before this component, the repository had an `LLMClient` abstraction and an `OpenAIClient` implementation, but there was no centralized mechanism for selecting the implementation based on configuration.

The application would therefore have needed to know which provider implementation to instantiate.

The architecture was changed to introduce a provider factory.

### 2.3 Architecture Before

    Application
         |
         v
    OpenAIClient
         |
         v
     OpenAI API

This creates unnecessary coupling between application code and a specific LLM provider.

### 2.4 Architecture After

    Application
         |
         v
    create_llm_client()
         |
         v
    LLMClient abstraction
         |
         +--------------------+
         |                    |
         v                    v
    OpenAIClient       Future Providers
         |
         v
     OpenAI API

The application interacts with `LLMClient` rather than directly constructing `OpenAIClient`.

### 2.5 Configuration Flow

Configuration is externalized through the existing `Settings` model.

The runtime flow is:

    Environment / .env
            |
            v
        Settings
            |
            v
      Provider Factory
            |
            v
       LLMClient
            |
            v
      Provider Client

The relevant configuration includes:

- `llm_provider`
- `openai_api_key`
- `llm_timeout_seconds`
- `llm_max_retries`
- `llm_initial_backoff_seconds`
- `llm_max_backoff_seconds`

### 2.6 Retry Configuration Integration

The existing `Settings` values were previously defined but the `OpenAIClient` was using the default `RetryPolicy()` values directly.

This component changed the OpenAI client so retry behavior is derived from application configuration.

The mapping is:

    llm_max_retries
            |
            v
    RetryPolicy.max_attempts = llm_max_retries + 1

The `+1` is intentional.

For example:

    llm_max_retries = 2

results in:

    Attempt 1 = initial request
    Attempt 2 = retry #1
    Attempt 3 = retry #2

Therefore:

    max_attempts = 3

Other configuration values are mapped directly:

    llm_initial_backoff_seconds
            -> RetryPolicy.initial_backoff_seconds

    llm_max_backoff_seconds
            -> RetryPolicy.max_backoff_seconds

    llm_timeout_seconds
            -> RetryPolicy.attempt_timeout_seconds

### 2.7 Provider Factory

A new provider factory was introduced:

    src/agent_platform/llm/factory.py

The factory exposes:

    create_llm_client(settings) -> LLMClient

Current provider selection:

    llm_provider = "openai"
            |
            v
       OpenAIClient

Provider names are normalized using lowercase conversion so values such as:

    openai
    OPENAI
    OpenAI

are treated consistently.

Unsupported providers result in:

    LLMConfigurationError

rather than an unclear runtime failure.

### 2.8 Architecture Decisions Made

#### Decision 1 — Application depends on LLMClient

Application/business logic should depend on:

    LLMClient

rather than:

    OpenAIClient

This follows dependency inversion and reduces provider coupling.

#### Decision 2 — Provider selection is configuration-driven

The provider is selected through:

    Settings.llm_provider

rather than hard-coded application logic.

#### Decision 3 — Provider-specific implementations remain isolated

Provider-specific SDK behavior remains inside provider implementations such as:

    OpenAIClient

The factory only selects the implementation.

#### Decision 4 — Unsupported providers fail fast

An unsupported provider configuration raises:

    LLMConfigurationError

This makes configuration problems explicit and easier to diagnose.

#### Decision 5 — Retry configuration is externalized

Retry behavior is controlled by `Settings` instead of hidden hard-coded values inside `OpenAIClient`.

### 2.9 Architecture Decisions Deliberately NOT Made

The following were intentionally not implemented yet:

- Anthropic implementation
- Azure OpenAI implementation
- Local model implementation
- Automatic provider failover
- Cross-provider load balancing
- Provider health scoring
- Dynamic provider routing
- Model selection service
- Multi-model routing
- Geographic provider routing

These are future architectural extensions.

The factory establishes the extension point without prematurely implementing unnecessary providers.

### 2.10 Technical Components Used

#### Pydantic Settings

Used for externalized application configuration.

Why:

- environment-based configuration
- type validation
- centralized configuration
- avoids hard-coded infrastructure settings

#### LLMClient abstraction

Used to define a provider-independent contract.

Why:

- provider abstraction
- dependency inversion
- easier testing
- future multi-provider support

#### Provider Factory

Used to centralize provider instantiation.

Why:

- removes provider construction from application logic
- creates a single provider-selection boundary
- enables future providers

#### RetryPolicy

Used to convert application configuration into runtime retry behavior.

Why:

- centralized resilience policy
- consistent retry behavior
- configurable timeout and retry limits

#### OpenAI Async Client

Used as the first concrete provider implementation.

Why:

- asynchronous API calls
- production-oriented integration
- compatible with the platform's async architecture

### 2.11 Testing Added

Three factory tests were added:

1. OpenAI provider creates `OpenAIClient`
2. Provider selection is case-insensitive
3. Unsupported provider raises `LLMConfigurationError`

One OpenAI configuration test was added:

1. Configured retry and timeout values are correctly propagated into `RetryPolicy`

The full test suite increased from:

    30 tests

to:

    34 tests

Final result:

    34 passed

### 2.12 Quality Gates

The component passed all repository quality gates:

    uv run ruff format --check .
    26 files already formatted

    uv run ruff check .
    All checks passed!

    uv run pytest
    34 passed

### 2.13 Interview Questions Addressed

This component prepares for questions such as:

**Q: How would you avoid vendor lock-in when building an enterprise LLM platform?**

Answer direction:

Use a provider-independent `LLMClient` abstraction and isolate provider-specific SDK implementations behind that abstraction.

**Q: How would you support multiple LLM providers?**

Answer direction:

Use configuration-driven provider selection through a factory. Each provider implements the same `LLMClient` contract.

**Q: Why use a factory instead of instantiating OpenAIClient directly?**

Answer direction:

The factory centralizes provider construction and prevents application/business logic from becoming coupled to a specific provider.

**Q: How would you change providers without modifying application logic?**

Answer direction:

Change configuration from one provider to another and provide an implementation of the same `LLMClient` interface.

**Q: Where should retry configuration live?**

Answer direction:

Retry policy should be externally configurable and injected into the provider client rather than hidden in provider-specific implementation defaults.

**Q: What is the difference between max retries and max attempts?**

Answer direction:

Max retries excludes the initial request. Max attempts includes the initial request.

Therefore:

    max_attempts = max_retries + 1

**Q: Why fail fast for unsupported providers?**

Answer direction:

Invalid infrastructure configuration should be detected during client construction rather than producing an ambiguous failure later during an LLM request.

### 2.14 Key Architectural Principle

The central principle introduced by this component is:

> Application logic should depend on stable platform abstractions, while provider-specific implementation details remain behind those abstractions.

This allows the LLM platform to evolve independently from application/business logic.

### 2.15 Current Layer 1 Architecture

At this point Layer 1 contains:

    Application
        |
        v
    LLMClient abstraction
        |
        v
    Provider Factory
        |
        v
    OpenAIClient
        |
        +--> Error Mapping
        |
        +--> Retry Policy
        |
        +--> Retry Executor
        |
        +--> Timeout
        |
        +--> Usage / Cost Metadata
        |
        v
    OpenAI API

This establishes the initial production-oriented LLM integration boundary.
