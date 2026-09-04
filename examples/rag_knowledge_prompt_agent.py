"""Retrieve an Email Router requirement from the product specification."""

import os
from pathlib import Path

from dotenv import load_dotenv

from workflow_agents.base_agents import RAGKnowledgePromptAgent


PRODUCT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "email_router"
    / "product_spec_email_router.txt"
)


def main() -> None:
    load_dotenv()

    agent = RAGKnowledgePromptAgent(
        os.getenv("OPENAI_BASE_URL"),
        os.environ["OPENAI_API_KEY"],
        (
            "You are an enterprise product analyst. Answer only from the retrieved "
            "product specification and state measurable targets precisely."
        ),
        os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        os.getenv("OPENAI_REASONING_EFFORT", "high"),
        chunk_size=2400,
        chunk_overlap=300,
    )
    product_specification = PRODUCT_SPEC_PATH.read_text(encoding="utf-8")
    agent.chunk_text(product_specification)
    agent.calculate_embeddings()

    prompt = (
        "What routing-accuracy target does the Email Router specify, and by when "
        "should it be achieved?"
    )
    print(agent.find_prompt_in_knowledge(prompt))


if __name__ == "__main__":
    main()
