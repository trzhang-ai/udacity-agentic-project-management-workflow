import os

from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

prompt = "What is the capital of France?"
persona = (
    "You are a college professor; your answers always start with: 'Dear students,'"
)

augmented_agent = AugmentedPromptAgent(
    openai_api_key, base_url, persona, "gpt-3.5-turbo"
)

augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)

# The agent uses general knowledge from the model's training. The system prompt
# guides the response to use the specified college-professor persona.
