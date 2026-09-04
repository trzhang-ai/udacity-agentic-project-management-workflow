# Architecture

## Purpose

The system converts an unstructured planning request and a product specification into three
linked artifact types: user stories, product features, and engineering tasks. The design
separates planning, routing, generation, evaluation, and state transfer so that each concern
can be replaced or tested independently.

## Runtime Flow

```text
Planning request
      |
      v
ActionPlanningAgent
      |
      v
RoutingAgent
      |
      +--> Product Manager worker --> evaluator --> user stories
      |
      +--> Program Manager worker --> evaluator --> product features
      |
      +--> Development worker -----> evaluator --> engineering tasks
                                                      |
                                                      v
                                           Consolidated plan
```

The routing step selects a role by comparing the embedding of the current action with the
embedding of each route description. The selected support function assembles a self-contained
work package containing the current action, the sample specification, and approved upstream
artifacts.

## Agent Responsibilities

### `ActionPlanningAgent`

Extracts an ordered set of executable steps from a planning request while staying within a
configured planning playbook.

### `RoutingAgent`

Uses cosine similarity over embeddings to select the specialist whose description most closely
matches the current step.

### `KnowledgeAugmentedPromptAgent`

Combines a role definition, bounded external knowledge, the current work package, and evaluator
feedback. Each call is stateless; all required context is carried in the request.

### `EvaluationAgent`

Runs a bounded worker/evaluator loop. It checks each candidate against role-specific format and
grounding criteria, then supplies corrective instructions to the worker when the answer is not
accepted.

### `RAGKnowledgePromptAgent`

Chunks a larger knowledge source, creates embeddings, retrieves the closest chunk, and uses only
that retrieved context when generating an answer.

## State and Grounding

The integrated Email Router workflow maintains three in-memory collections:

- approved user stories
- approved product features
- approved engineering tasks

Downstream work packages include the relevant upstream collections. This prevents the routing
decision from becoming the only information passed between agents and keeps generated features
and tasks traceable to earlier artifacts.

## Failure Handling

- The action planner must return the expected number of stages.
- Feature generation cannot run before user stories exist.
- Engineering-task generation cannot run before both stories and features exist.
- Evaluator loops have a maximum interaction count.
- The workflow verifies that every required artifact collection is populated before consolidation.

## Current Boundaries

The Email Router document is a sample input, not an implemented application. Its KPI values are
planning targets, not observed outcomes. The prototype uses in-memory orchestration and free-form
text contracts; production hardening would add structured response schemas, persistent state,
telemetry, automated evaluation sets, and human approval controls.
