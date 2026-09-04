"""Route operational emails to specialized policy-grounded agents."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from workflow_agents.base_agents import RoutingAgent


def main() -> None:
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "high")

    compliance_agent = KnowledgeAugmentedPromptAgent(
        base_url,
        api_key,
        "You are a privacy and compliance operations specialist.",
        (
            "Handle data deletion, privacy, legal, and suspected breach requests. "
            "Require specialist review and never authorize an automated send."
        ),
        model,
        reasoning_effort,
    )
    billing_agent = KnowledgeAugmentedPromptAgent(
        base_url,
        api_key,
        "You are a billing operations specialist.",
        (
            "Handle invoices, duplicate charges, payment status, refunds, and "
            "account-credit inquiries."
        ),
        model,
        reasoning_effort,
    )
    integration_agent = KnowledgeAugmentedPromptAgent(
        base_url,
        api_key,
        "You are an email-platform integration specialist.",
        (
            "Handle SMTP, IMAP, API, authentication, webhook, and email-ingestion "
            "failures."
        ),
        model,
        reasoning_effort,
    )

    agents = [
        {
            "name": "privacy and compliance",
            "description": (
                "Data rights, privacy, legal requests, retention, or suspected "
                "security breaches"
            ),
            "func": lambda prompt: compliance_agent.respond(prompt),
        },
        {
            "name": "billing operations",
            "description": (
                "Invoices, charges, payments, refunds, and account credits"
            ),
            "func": lambda prompt: billing_agent.respond(prompt),
        },
        {
            "name": "platform integrations",
            "description": (
                "SMTP, IMAP, APIs, authentication, webhooks, and message ingestion"
            ),
            "func": lambda prompt: integration_agent.respond(prompt),
        },
    ]
    router = RoutingAgent(base_url, api_key, agents)

    messages = (
        "Please delete every copy of my personal data and confirm completion.",
        "Our latest invoice includes the same annual subscription charge twice.",
        "The IMAP connector stopped ingesting messages after credential rotation.",
    )
    for message in messages:
        print(f"\nIncoming email: {message}")
        print(router.route(message))


if __name__ == "__main__":
    main()
