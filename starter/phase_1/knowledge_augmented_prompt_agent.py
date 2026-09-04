# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

prompt = "What is the capital of France?"

persona = "a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"
aug_prompt_agent = KnowledgeAugmentedPromptAgent(
    base_url, openai_api_key, persona, knowledge, "gpt-5-nano", "medium"
)

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
aug_prompt_agent_response = aug_prompt_agent.respond(prompt)
print(aug_prompt_agent_response)
