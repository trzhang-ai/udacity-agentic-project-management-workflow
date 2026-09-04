"""Evaluate a policy-grounded Email Router handling decision."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import EvaluationAgent
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

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "high")

    worker = KnowledgeAugmentedPromptAgent(
        base_url,
        api_key,
        (
            "You are an email-operations policy analyst. State the handling "
            "decision first, followed by one concise rationale."
        ),
        ROUTING_POLICY,
        model,
        reasoning_effort,
    )
    evaluator = EvaluationAgent(
        base_url,
        api_key,
        model,
        reasoning_effort,
        (
            "You are a quality-control reviewer for an enterprise email-routing "
            "system. Reject answers that are ungrounded, ambiguous, or operationally "
            "unsafe."
        ),
        (
            "The answer must route the message to human review, cite the 0.82 "
            "confidence score and the applicable 0.70-0.89 policy band, and avoid "
            "claiming that the response may be sent automatically."
        ),
        worker,
        max_interactions=4,
    )
    prompt = (
        "A routine account-access email received a classification confidence score "
        "of 0.82. How should the Email Router handle it?"
    )

    print(evaluator.evaluate(prompt))


if __name__ == "__main__":
    main()
