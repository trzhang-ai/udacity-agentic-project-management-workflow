"""Send a focused Email Router operations question directly to the model."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import DirectPromptAgent


def main() -> None:
    load_dotenv()

    agent = DirectPromptAgent(
        os.environ["OPENAI_API_KEY"],
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
    )
    prompt = (
        "Identify the three most important operational controls for an AI system "
        "that classifies inbound customer email and drafts routine responses."
    )

    print(agent.respond(prompt))


if __name__ == "__main__":
    main()
