# Test script for DirectPromptAgent class

# TODO: 1 - Import the DirectPromptAgent class from BaseAgents
from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_prompt_agent = DirectPromptAgent(openai_api_key, base_url, "gpt-5.6-luna")

# TODO: 4 - Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_prompt_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# TODO: 5 - Print an explanatory message describing the knowledge source used by the agent to generate the response
print("The agent uses general knowledge from base model training")
