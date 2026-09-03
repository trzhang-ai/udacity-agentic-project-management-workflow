# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import RoutingAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent


# Load environment variables from .env file
load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
texas_expert = KnowledgeAugmentedPromptAgent(
    base_url, openai_api_key, persona, knowledge, "gpt-5-nano", "medium"
)

knowledge = "You know everything about Europe"
# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
europe_agent = KnowledgeAugmentedPromptAgent(
    base_url, openai_api_key, persona, knowledge, "gpt-5-nano", "medium"
)

persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
math_expert = KnowledgeAugmentedPromptAgent(
    base_url, openai_api_key, persona, knowledge, "gpt-5-nano", "medium"
)

routing_agent = RoutingAgent(base_url, openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        # TODO: 5 - Call the Texas Agent to respond to prompts
        "func": lambda x: texas_expert.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        # TODO: 6 - Define a function to call the Europe Agent
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        # TODO: 7 - Define a function to call the Math Agent
        "func": lambda x: math_expert.respond(x),
    },
]

routing_agent.agents = agents

# TODO: 8 - Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"

q1 = "Tell me about the history of Rome, Texas"
print(f"{q1}: {routing_agent.route(q1)}")
q2 = "Tell me about the history of Rome, Italy"
print(f"{q2}: {routing_agent.route(q2)}")
q3 = "One story takes 2 days, and there are 20 stories"
print(f"{q3}: {routing_agent.route(q3)}")
