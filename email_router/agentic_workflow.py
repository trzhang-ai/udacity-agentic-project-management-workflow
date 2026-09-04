"""Build a traceable development plan from the Email Router specification."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from workflow_agents.base_agents import ActionPlanningAgent
from workflow_agents.base_agents import EvaluationAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from workflow_agents.base_agents import RoutingAgent


DEFAULT_MODEL_NAME = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_WORKFLOW_PROMPT = "What would the development tasks for this product be?"
MAX_EVALUATION_INTERACTIONS = 10
PRODUCT_SPEC_PATH = Path(__file__).with_name("product_spec_email_router.txt")


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime settings for the model endpoint."""

    base_url: str
    api_key: str = field(repr=False)
    model_name: str = DEFAULT_MODEL_NAME
    reasoning_effort: str = DEFAULT_REASONING_EFFORT


def load_configuration() -> WorkflowConfig:
    """Load and validate model configuration without exposing credentials."""

    load_dotenv()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model_name = (os.getenv("OPENAI_MODEL") or "").strip() or DEFAULT_MODEL_NAME
    reasoning_effort = (
        (os.getenv("OPENAI_REASONING_EFFORT") or "").strip()
        or DEFAULT_REASONING_EFFORT
    )

    required_values = {
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment configuration: " + ", ".join(missing)
        )

    return WorkflowConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )


def load_product_spec() -> str:
    """Read the product specification and reject an empty document."""

    try:
        product_spec = PRODUCT_SPEC_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load the product specification at {PRODUCT_SPEC_PATH}."
        ) from exc

    if not product_spec:
        raise RuntimeError(f"Product specification is empty: {PRODUCT_SPEC_PATH}")
    return product_spec


def combine_outputs(outputs: list[str]) -> str:
    """Join approved artifacts for inclusion in downstream work packages."""

    return "\n\n".join(outputs) if outputs else "(none yet)"


class EmailRouterPlanningWorkflow:
    """Coordinate planning, routing, specialist generation, and evaluation."""

    def __init__(self, config: WorkflowConfig, product_spec: str):
        self.product_spec = product_spec
        self.workflow_state: dict[str, list[str]] = {
            "user_stories": [],
            "features": [],
            "engineering_tasks": [],
        }

        planning_knowledge = (
            "Stories are defined from a product specification by identifying a "
            "persona, an action, and a desired outcome for each story. "
            "Each story represents a specific product capability.\n"
            "Features group related user stories.\n"
            "Engineering tasks describe the work required to implement each story.\n"
            "A development plan contains all three artifact types.\n"
            "Return exactly three plain-text steps in this order, with one step per "
            "line and no heading, numbering, bullets, explanations, or blank lines:\n"
            "Create user stories from the product specification\n"
            "Group those user stories into product features\n"
            "Create engineering tasks for those user stories and features\n"
            "Do not create separate subtask, testing, deployment, or release-planning "
            "steps."
        )
        self.action_planning_agent = ActionPlanningAgent(
            config.base_url,
            config.api_key,
            planning_knowledge,
            config.model_name,
            config.reasoning_effort,
        )

        grounding_criteria = (
            "The answer must be grounded in the Email Router product specification and "
            "workflow artifacts supplied in the current work package. Reject unrelated "
            "products, unsupported capabilities, placeholder examples, and requests for "
            "information already supplied."
        )
        evaluator_persona = (
            "You evaluate worker-agent output against explicit structure and grounding "
            "requirements."
        )

        product_manager_worker = KnowledgeAugmentedPromptAgent(
            config.base_url,
            config.api_key,
            (
                "You are a Product Manager responsible for defining product user "
                "stories."
            ),
            (
                "Write user stories with a persona, action, and desired outcome. "
                "Each story begins with 'As a'. Use the user classes and capabilities "
                "in this product specification:\n\n"
                + product_spec
            ),
            config.model_name,
            config.reasoning_effort,
        )
        self.product_manager_evaluator = EvaluationAgent(
            config.base_url,
            config.api_key,
            config.model_name,
            config.reasoning_effort,
            evaluator_persona,
            (
                "Every story must follow this structure: As a [type of user], I want "
                "[an action or feature] so that [benefit/value].\n" + grounding_criteria
            ),
            product_manager_worker,
            MAX_EVALUATION_INTERACTIONS,
        )

        program_manager_worker = KnowledgeAugmentedPromptAgent(
            config.base_url,
            config.api_key,
            (
                "You are a Program Manager responsible for organizing user stories "
                "into product features."
            ),
            (
                "Define cohesive product features by grouping related user stories. "
                "Preserve the intent and traceability of the supplied stories."
            ),
            config.model_name,
            config.reasoning_effort,
        )
        self.program_manager_evaluator = EvaluationAgent(
            config.base_url,
            config.api_key,
            config.model_name,
            config.reasoning_effort,
            evaluator_persona,
            (
                "Every feature must use these exact field labels:\n"
                "Feature Name: A clear, concise title that identifies the capability\n"
                "Description: A brief explanation of what the feature does and its "
                "purpose\n"
                "Key Functionality: The specific capabilities or actions the feature "
                "provides\n"
                "User Benefit: How this feature creates value for the user\n"
                "A heading or unlabeled title does not replace the Feature Name field. "
                "Each feature must group one or more user stories supplied in the "
                "current work package.\n"
                + grounding_criteria
            ),
            program_manager_worker,
            MAX_EVALUATION_INTERACTIONS,
        )

        development_engineer_worker = KnowledgeAugmentedPromptAgent(
            config.base_url,
            config.api_key,
            (
                "You are a Development Engineer responsible for defining implementation "
                "tasks."
            ),
            (
                "Define the engineering work required to implement each supplied user "
                "story and feature. Preserve explicit traceability to the approved "
                "planning artifacts."
            ),
            config.model_name,
            config.reasoning_effort,
        )
        self.development_engineer_evaluator = EvaluationAgent(
            config.base_url,
            config.api_key,
            config.model_name,
            config.reasoning_effort,
            evaluator_persona,
            (
                "Every task must use these exact field labels:\n"
                "Task ID: A unique identifier for tracking purposes\n"
                "Task Title: A brief description of the development work\n"
                "Related User Story: A reference to the parent user story\n"
                "Description: A detailed explanation of the technical work required\n"
                "Acceptance Criteria: Specific requirements for completion\n"
                "Estimated Effort: A time or complexity estimate\n"
                "Dependencies: Tasks that must be completed first\n"
                "A heading or Title field does not replace the Task Title field. Every "
                "task must implement a supplied Email Router user story or feature. "
                "Every Related User Story field must quote or clearly identify a story "
                "from the current work package rather than inventing an absent ID.\n"
                + grounding_criteria
            ),
            development_engineer_worker,
            MAX_EVALUATION_INTERACTIONS,
        )

        routes = [
            {
                "name": "Product Manager",
                "description": (
                    "Identifies product user personas and writes user stories in the "
                    "'As a..., I want..., so that...' format."
                ),
                "func": self.product_manager_support_function,
            },
            {
                "name": "Program Manager",
                "description": (
                    "Groups related user stories into product features with a name, "
                    "description, key functionality, and user benefit."
                ),
                "func": self.program_manager_support_function,
            },
            {
                "name": "Development Engineer",
                "description": (
                    "Converts approved user stories and product features into engineering "
                    "tasks with acceptance criteria, effort, and dependencies."
                ),
                "func": self.development_engineer_support_function,
            },
        ]
        self.routing_agent = RoutingAgent(
            config.base_url,
            config.api_key,
            routes,
        )

    def reset_state(self) -> None:
        """Clear artifacts so repeated runs start from an isolated workflow state."""

        for outputs in self.workflow_state.values():
            outputs.clear()

    def product_manager_support_function(self, step: str) -> str:
        work_package = f"""
Current step:
{step}

Product specification:
{self.product_spec}

Previously completed user-story work:
{combine_outputs(self.workflow_state["user_stories"])}

Complete only the current step.
Produce Email Router user stories grounded in the product specification.
Do not create an unrelated example product.
""".strip()

        # The evaluator owns generation as well as validation for its worker.
        result = self.product_manager_evaluator.evaluate(work_package)
        final_response = result["final_response"]
        self.workflow_state["user_stories"].append(final_response)
        return final_response

    def program_manager_support_function(self, step: str) -> str:
        if not self.workflow_state["user_stories"]:
            raise RuntimeError(
                "Features cannot be created before user stories have been completed."
            )

        work_package = f"""
Current step:
{step}

Product specification:
{self.product_spec}

Approved user stories:
{combine_outputs(self.workflow_state["user_stories"])}

Previously completed feature work:
{combine_outputs(self.workflow_state["features"])}

Complete only the current step.
Group the supplied Email Router user stories into product features.
For every feature, use these exact field labels in this order:
Feature Name:
Description:
Key Functionality:
User Benefit:
A heading alone does not replace the Feature Name field.
Do not invent a different backlog or ask for stories already supplied here.
""".strip()

        result = self.program_manager_evaluator.evaluate(work_package)
        final_response = result["final_response"]
        self.workflow_state["features"].append(final_response)
        return final_response

    def development_engineer_support_function(self, step: str) -> str:
        if not self.workflow_state["user_stories"] or not self.workflow_state["features"]:
            raise RuntimeError(
                "Engineering tasks require completed user stories and features."
            )

        work_package = f"""
Current step:
{step}

Product specification:
{self.product_spec}

Approved user stories:
{combine_outputs(self.workflow_state["user_stories"])}

Approved product features:
{combine_outputs(self.workflow_state["features"])}

Previously completed engineering work:
{combine_outputs(self.workflow_state["engineering_tasks"])}

Complete only the current step.
Create engineering work for the supplied Email Router stories and features.
For every task, use these exact field labels in this order:
Task ID:
Task Title:
Related User Story:
Description:
Acceptance Criteria:
Estimated Effort:
Dependencies:
Do not replace Task Title with Title or an unlabeled heading.
In each Related User Story field, quote or clearly identify the corresponding
supplied story; do not invent a story ID absent from the supplied stories.
Do not invent an unrelated example product.
""".strip()

        result = self.development_engineer_evaluator.evaluate(work_package)
        final_response = result["final_response"]
        self.workflow_state["engineering_tasks"].append(final_response)
        return final_response

    def run(self, workflow_prompt: str = DEFAULT_WORKFLOW_PROMPT) -> str:
        """Execute the three-stage planning pipeline and return its consolidated plan."""

        self.reset_state()
        print("\n*** Workflow execution started ***\n")
        print(f"Planning request: {workflow_prompt}")
        print("\nDefining workflow steps")

        workflow_steps = self.action_planning_agent.extract_steps_from_prompt(
            workflow_prompt
        )
        if len(workflow_steps) != 3:
            raise RuntimeError(
                "Expected exactly three workflow steps, but received "
                f"{len(workflow_steps)}: {workflow_steps}"
            )

        completed_steps = []
        for step in workflow_steps:
            print(f"Executing: {step}")
            result = self.routing_agent.route(step)
            completed_steps.append(result)
            print(f"Result:\n{result}")

        missing_artifacts = [
            name for name, outputs in self.workflow_state.items() if not outputs
        ]
        if missing_artifacts:
            raise RuntimeError(
                "Workflow finished without these required artifacts: "
                + ", ".join(missing_artifacts)
            )

        final_output = f"""
EMAIL ROUTER DEVELOPMENT PLAN

USER STORIES
{combine_outputs(self.workflow_state["user_stories"])}

PRODUCT FEATURES
{combine_outputs(self.workflow_state["features"])}

ENGINEERING TASKS
{combine_outputs(self.workflow_state["engineering_tasks"])}
""".strip()

        print("\n*** Final consolidated output ***\n")
        print(final_output)
        return final_output


def main() -> None:
    """Load runtime dependencies and execute the integrated planning workflow."""

    config = load_configuration()
    product_spec = load_product_spec()
    workflow = EmailRouterPlanningWorkflow(config, product_spec)
    workflow.run()


if __name__ == "__main__":
    main()
