"""Apply an operations-lead persona to an Email Router rollout brief."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent


def main() -> None:
    load_dotenv()

    persona = (
        "You are an enterprise operations lead. Write for senior stakeholders, "
        "separate decisions from risks, and end with measurable next actions."
    )
    prompt = (
        "Prepare a concise executive brief for a 30-day Email Router pilot in "
        "customer support. Cover rollout sequencing, human oversight, and the "
        "metrics required for a go-or-no-go decision."
    )
    agent = AugmentedPromptAgent(
        os.environ["OPENAI_API_KEY"],
        os.getenv("OPENAI_BASE_URL"),
        persona,
        os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
    )

    print(agent.respond(prompt))


if __name__ == "__main__":
    main()
