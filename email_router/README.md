# Email Router Planning Workflow

This directory contains a prototype case study for converting a product specification into a
structured development plan. KPI targets in the sample specification are illustrative planning
assumptions, not observed business results.

## Files

- `agentic_workflow.py` configures the planner, specialist workers, evaluators, router, and
  workflow state.
- `product_spec_email_router.txt` is the sample product specification supplied to each relevant
  work package.

## Role Pipeline

- Product Manager: derives user stories from the specification.
- Program Manager: groups approved stories into product features.
- Development Engineer: converts approved stories and features into implementation tasks.

Each worker is paired with an evaluator. Approved output becomes explicit state for the next
role, preserving traceability across the pipeline.

Run the workflow from the repository root:

```bash
uv run python -m email_router.agentic_workflow
```

The generated output is planning material and should receive human review before it is used for
delivery commitments.
