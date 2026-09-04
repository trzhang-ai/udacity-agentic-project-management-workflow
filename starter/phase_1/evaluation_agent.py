import os

from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
prompt = "What is the capital of France?"

# Parameters for the knowledge agent
worker_persona = (
    "You are a college professor; your answer always starts with: Dear students,"
)
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    worker_persona,
    knowledge,
    "gpt-3.5-turbo",
    None,
)

# Parameters for the evaluation agent
evaluator_persona = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    base_url,
    openai_api_key,
    "gpt-3.5-turbo",
    None,
    evaluator_persona,
    evaluation_criteria,
    knowledge_agent,
    max_interactions=10,
)

print(evaluation_agent.evaluate(prompt))
