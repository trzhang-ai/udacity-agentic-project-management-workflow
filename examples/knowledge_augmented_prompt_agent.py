"""Apply a bounded operating policy to an Email Router decision."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent


ROUTING_POLICY = """
Email Router confidence policy:
- Confidence of 0.90 or higher: create an automated draft for agent approval.
- Confidence from 0.70 through 0.89: send the message to human review.
- Confidence below 0.70: place the message in manual triage.
- Security, legal, and data-rights requests always require specialist review.
""".strip()


def main() -> None:
    load_dotenv()

    agent = KnowledgeAugmentedPromptAgent(
        os.getenv("OPENAI_BASE_URL"),
        os.environ["OPENAI_API_KEY"],
        (
            "You are an email-operations policy analyst. State the handling "
            "decision first and explain it in one sentence."
        ),
        ROUTING_POLICY,
        os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        os.getenv("OPENAI_REASONING_EFFORT", "high"),
    )
    prompt = (
        "A routine account-access email received a classification confidence score "
        "of 0.82. How should the Email Router handle it?"
    )

    print(agent.respond(prompt))


if __name__ == "__main__":
    main()
