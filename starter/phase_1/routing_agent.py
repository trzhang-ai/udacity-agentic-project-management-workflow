import os

from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from workflow_agents.base_agents import RoutingAgent


# Load environment variables from .env file
load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

texas_knowledge = "You know everything about Texas"
texas_expert = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    persona,
    texas_knowledge,
    "gpt-3.5-turbo",
    None,
)

europe_knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    persona,
    europe_knowledge,
    "gpt-3.5-turbo",
    None,
)

math_persona = "You are a college math professor"
math_knowledge = (
    "You know everything about math. You take prompts with numbers, extract "
    "math formulas, and show the answer without explanation."
)
math_expert = KnowledgeAugmentedPromptAgent(
    base_url,
    openai_api_key,
    math_persona,
    math_knowledge,
    "gpt-3.5-turbo",
    None,
)

routing_agent = RoutingAgent(base_url, openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_expert.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_expert.respond(x),
    },
]

routing_agent.agents = agents

texas_prompt = "Tell me about the history of Rome, Texas"
print(f"{texas_prompt}: {routing_agent.route(texas_prompt)}")
europe_prompt = "Tell me about the history of Rome, Italy"
print(f"{europe_prompt}: {routing_agent.route(europe_prompt)}")
math_prompt = "One story takes 2 days, and there are 20 stories"
print(f"{math_prompt}: {routing_agent.route(math_prompt)}")
