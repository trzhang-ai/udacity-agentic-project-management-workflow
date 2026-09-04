import os

from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

prompt = "What is the capital of France?"

persona = "You are a college professor; your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    base_url, openai_api_key, persona, knowledge, "gpt-3.5-turbo", None
)

knowledge_agent_response = knowledge_agent.respond(prompt)
print(knowledge_agent_response)
print(
    "The answer uses the supplied knowledge, which states that London, not Paris, "
    "is the capital of France."
)
