# agentic_workflow.py
import os
from dotenv import load_dotenv
from pathlib import Path

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from workflow_agents.base_agents import EvaluationAgent
from workflow_agents.base_agents import RoutingAgent


# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-5.6-luna"
REASONING_EFFORT = "high"

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
product_spec = (
    Path(__file__)
    .with_name("Product-Spec-Email-Router.txt")
    .read_text(encoding="utf-8")
)

# Instantiate all the agents
# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development plan for a product contains all these components.\n"
    "Return exactly three plain-text steps in this order, with one step per line "
    "and no heading, numbering, bullets, explanations, or blank lines:\n"
    "Create user stories from the product specification\n"
    "Group those user stories into product features\n"
    "Create engineering tasks for those user stories and features\n"
    "Do not create separate subtask, testing, deployment, or release-planning steps."
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    base_url,
    openai_api_key,
    knowledge_action_planning,
    MODEL_NAME,
    REASONING_EFFORT,
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    + product_spec
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager,
    MODEL_NAME,
    REASONING_EFFORT,
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona = "You are an evaluation agent that checks the answers of other worker agents"
grounding_criteria = (
    "The answer must be grounded in the Email Router product specification and "
    "workflow artifacts supplied in the current work package. Reject unrelated "
    "products, unsupported capabilities, placeholder examples, and requests for "
    "information already supplied."
)
evaluation_criteria = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value].\n"
    + grounding_criteria
)
product_manager_evaluation_agent = EvaluationAgent(
    base_url,
    openai_api_key,
    MODEL_NAME,
    REASONING_EFFORT,
    persona,
    evaluation_criteria,
    product_manager_knowledge_agent,
    10,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager,
    MODEL_NAME,
    REASONING_EFFORT,
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria = (
    "The answer must contain product features using all of these exact field labels "
    "for every feature:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user\n"
    "A heading or an unlabeled title is not a substitute for the exact "
    "Feature Name: field. If any required field is omitted or renamed, the answer "
    "must be evaluated as No.\n"
    "Each feature must group one or more user stories supplied in the current "
    "work package.\n"
    + grounding_criteria
)

program_manager_evaluation_agent = EvaluationAgent(
    base_url,
    openai_api_key,
    MODEL_NAME,
    REASONING_EFFORT,
    persona_program_manager_eval,
    evaluation_criteria,
    program_manager_knowledge_agent,
    10,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer,
    MODEL_NAME,
    REASONING_EFFORT,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria = (
    "The answer must contain tasks using all of these exact field labels for every "
    "task:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first\n"
    "The label Title: or a heading is not a substitute for the exact Task Title: "
    "field. If any required field is omitted or renamed, the answer must be "
    "evaluated as No.\n"
    "Every task must implement a supplied Email Router user story or feature. "
    "Every Related User Story field must quote or clearly identify a user story "
    "supplied in the current work package rather than inventing an absent story ID.\n"
    + grounding_criteria
)
development_engineer_evaluation_agent = EvaluationAgent(
    base_url,
    openai_api_key,
    MODEL_NAME,
    REASONING_EFFORT,
    persona_dev_engineer_eval,
    evaluation_criteria,
    development_engineer_knowledge_agent,
    10,
)


# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(base_url, openai_api_key, {})

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

workflow_state = {
    "user_stories": [],
    "features": [],
    "engineering_tasks": [],
}


def combine_outputs(outputs):
    return "\n\n".join(outputs) if outputs else "(none yet)"


def product_manager_support_function(step):
    work_package = f"""
Current step:
{step}

Product specification:
{product_spec}

Previously completed user-story work:
{combine_outputs(workflow_state["user_stories"])}

Complete only the current step.
Produce Email Router user stories grounded in the product specification.
Do not create an unrelated example product.
""".strip()

    # EvaluationAgent.evaluate() already asks its worker agent for a response.
    result = product_manager_evaluation_agent.evaluate(work_package)
    final_response = result["final_response"]
    workflow_state["user_stories"].append(final_response)
    return final_response


def program_manager_support_function(step):
    if not workflow_state["user_stories"]:
        raise RuntimeError(
            "Features cannot be created before user stories have been completed."
        )

    work_package = f"""
Current step:
{step}

Product specification:
{product_spec}

Approved user stories:
{combine_outputs(workflow_state["user_stories"])}

Previously completed feature work:
{combine_outputs(workflow_state["features"])}

Complete only the current step.
Group the supplied Email Router user stories into product features.
For every feature, use these exact field labels in this order:
Feature Name:
Description:
Key Functionality:
User Benefit:
A heading alone does not replace the Feature Name: field.
Do not invent a different backlog or ask for stories already supplied here.
""".strip()

    result = program_manager_evaluation_agent.evaluate(work_package)
    final_response = result["final_response"]
    workflow_state["features"].append(final_response)
    return final_response


def development_engineer_support_function(step):
    if not workflow_state["user_stories"] or not workflow_state["features"]:
        raise RuntimeError(
            "Engineering tasks require completed user stories and features."
        )

    work_package = f"""
Current step:
{step}

Product specification:
{product_spec}

Approved user stories:
{combine_outputs(workflow_state["user_stories"])}

Approved product features:
{combine_outputs(workflow_state["features"])}

Previously completed engineering work:
{combine_outputs(workflow_state["engineering_tasks"])}

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
Do not replace Task Title: with Title: or with a heading.
If the step requests subtasks, refine the previously completed tasks.
In each Related User Story field, quote or clearly identify the corresponding
supplied As a... story; do not invent a story ID absent from the supplied stories.
Do not invent an unrelated example product.
""".strip()

    result = development_engineer_evaluation_agent.evaluate(work_package)
    final_response = result["final_response"]
    workflow_state["engineering_tasks"].append(final_response)
    return final_response

agents = [
    {
        "name": "Product Manager",
        "description": (
            "Identifies product user personas and writes user stories "
            "in the 'As a..., I want..., so that...' format."
        ),
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": (
            "Groups related user stories into product features. Produces "
            "Feature Name, Description, Key Functionality, and User Benefit."
        ),
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Converts user stories and product features into engineering tasks "
            "with task IDs, acceptance criteria, effort, and dependencies."
        ),
        "func": development_engineer_support_function,
    },
]
routing_agent.agents = agents

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
if len(workflow_steps) != 3:
    raise RuntimeError(
        "Expected exactly three workflow steps, but received "
        f"{len(workflow_steps)}: {workflow_steps}"
    )

completed_steps = []
for step in workflow_steps:
    print(f"Executing {step} ...")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"Result: {result}")

if not completed_steps:
    raise RuntimeError("The action planning agent returned no workflow steps.")

missing_artifacts = [name for name, outputs in workflow_state.items() if not outputs]
if missing_artifacts:
    raise RuntimeError(
        "Workflow finished without these required artifacts: "
        + ", ".join(missing_artifacts)
    )

# Retain the starter's required last-completed-step output.
print(completed_steps[-1])

final_output = f"""
EMAIL ROUTER DEVELOPMENT PLAN

USER STORIES
{combine_outputs(workflow_state["user_stories"])}

PRODUCT FEATURES
{combine_outputs(workflow_state["features"])}

ENGINEERING TASKS
{combine_outputs(workflow_state["engineering_tasks"])}
""".strip()

print("\n*** Final consolidated output ***\n")
print(final_output)
