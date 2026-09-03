# TODO: 1 - Import the AugmentedPromptAgent class
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

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
aug_prompt_agent = AugmentedPromptAgent(openai_api_key, base_url, persona, "gpt-5-nano")


# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
aug_agent_response = aug_prompt_agent.respond(prompt)

# Print the agent's response
print(aug_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# General knowledge from base model training + person in system prompt
# - How the system prompt specifying the persona affected the agent's response.
# The LLM will nudge its response to align with the requirements of the system prompt.
