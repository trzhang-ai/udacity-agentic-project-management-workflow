"""Build an incident-response plan from an approved Email Router runbook."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import ActionPlanningAgent


INCIDENT_RUNBOOK = """
Email Router degraded-confidence response plan
1. Pause automatic email sending while leaving ingestion enabled.
2. Route all messages with confidence below 0.90 to the human-review queue.
3. Notify the on-call operations lead and the model owner.
4. Compare current category-level confidence with the previous 24-hour baseline.
5. Sample recent classifications and record confirmed routing errors.
6. Roll back the latest model or rules change if it caused the degradation.
7. Resume automatic sending only after confidence and routing accuracy recover.
8. Publish an incident summary with impact, root cause, and follow-up actions.
""".strip()


def main() -> None:
    load_dotenv()

    agent = ActionPlanningAgent(
        os.getenv("OPENAI_BASE_URL"),
        os.environ["OPENAI_API_KEY"],
        INCIDENT_RUNBOOK,
        os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        os.getenv("OPENAI_REASONING_EFFORT", "high"),
    )
    incident = (
        "Classification confidence dropped sharply after a routing-rules release. "
        "Prepare the approved operational response plan."
    )

    for step in agent.extract_steps_from_prompt(incident):
        print(f"- {step}")


if __name__ == "__main__":
    main()
