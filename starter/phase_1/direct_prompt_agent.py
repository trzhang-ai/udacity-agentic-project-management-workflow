import os

from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

prompt = "What is the capital of France?"

direct_agent = DirectPromptAgent(openai_api_key, base_url, "gpt-3.5-turbo")

direct_agent_response = direct_agent.respond(prompt)

print(direct_agent_response)
print("The agent used general knowledge from the selected model's training.")
