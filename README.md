# Agentic Project Planning Workflow

A Python prototype that turns a product specification and planning request into
traceable user stories, feature definitions, and engineering tasks. The pipeline
decomposes the request, routes each step to a role-specific agent, and applies
evaluator-driven revision before returning a structured development plan.

The bundled Email Router specification is a sample case study. This repository
implements the planning workflow; it does not implement or deploy an email-routing
service, and the targets in the specification are illustrative rather than measured
business results.

## Capabilities

- Reusable direct, persona-augmented, knowledge-grounded, and retrieval-augmented agents
- Semantic routing through embedding similarity
- Action-plan decomposition for multi-stage work
- Worker/evaluator loops with bounded revision cycles
- Explicit state transfer between product, program, and engineering roles
- Consolidated user-story, feature, and engineering-task output

## Architecture

The integrated workflow follows this sequence:

1. `ActionPlanningAgent` converts the request into three ordered planning steps.
2. `RoutingAgent` selects the Product Manager, Program Manager, or Development Engineer route.
3. A knowledge-grounded worker produces the artifact for that step.
4. `EvaluationAgent` checks structure and grounding, then requests a revision when needed.
5. Approved outputs are added to workflow state and supplied to downstream roles.
6. The workflow consolidates all approved artifacts into one development plan.

See [the architecture notes](docs/architecture.md) for component responsibilities,
state flow, and design limitations.

## Repository Layout

```text
workflow_agents/     Reusable agent implementations
email_router/        Integrated Email Router planning workflow and sample specification
examples/            Focused component smoke examples
tests/               Offline unit and orchestration tests
docs/                Architecture and design documentation
```

## Configuration

Python 3.13 and [`uv`](https://docs.astral.sh/uv/) are used for dependency management.

```bash
uv sync --locked
cp .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env`. The model and reasoning-effort values
below are the defaults and must be supported by the selected OpenAI-compatible endpoint:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_MODEL=gpt-5.6-luna
OPENAI_REASONING_EFFORT=high
```

Secrets are loaded at runtime and excluded from version control.

## Run the Integrated Workflow

From the repository root:

```bash
uv run python -m email_router.agentic_workflow
```

The command prints routing decisions, evaluator verdicts, intermediate artifacts, and the
final consolidated plan.

## Run Component Examples

Each example exercises one reusable agent in a business-oriented scenario:

```bash
uv run python -m examples.direct_prompt_agent
uv run python -m examples.augmented_prompt_agent
uv run python -m examples.knowledge_augmented_prompt_agent
uv run python -m examples.rag_knowledge_prompt_agent
uv run python -m examples.evaluation_agent
uv run python -m examples.routing_agent
uv run python -m examples.action_planning_agent
```

The examples call the configured endpoint. The offline test suite covers core routing,
evaluation, retrieval, and workflow-state behavior without making network requests.

## Run Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Design Boundaries

- Generation quality depends on the configured model and endpoint.
- Evaluator approval improves consistency but does not prove factual correctness.
- Embedding-based routing is intentionally lightweight and should be calibrated with a
  representative evaluation set before production use.
- The prototype has no persistent workflow store, retry queue, observability backend, or
  human approval interface.
- Production use would require broader regression and evaluation coverage, structured schemas,
  authentication controls, cost and latency monitoring, and data-governance review.
