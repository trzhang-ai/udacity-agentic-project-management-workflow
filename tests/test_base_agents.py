"""Offline behavioral tests for reusable agent components."""

import csv
from contextlib import chdir, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from workflow_agents.base_agents import EvaluationAgent
from workflow_agents.base_agents import RAGKnowledgePromptAgent
from workflow_agents.base_agents import RoutingAgent


def completion_response(content):
    """Build the response shape consumed by the agent implementations."""

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RAGKnowledgePromptAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = RAGKnowledgePromptAgent(
            base_url="https://example.invalid/v1",
            openai_api_key="test-key",
            persona="Operations knowledge analyst",
            model_name="test-model",
            reasoning_effort="low",
            chunk_size=256,
            chunk_overlap=16,
        )

    def test_short_text_is_persisted_as_one_chunk(self):
        source_text = "Escalate low-confidence email classifications for human review."

        with TemporaryDirectory() as directory, chdir(directory):
            chunks = self.agent.chunk_text(source_text)
            persisted_path = self.agent.chunks_path

            self.assertTrue(persisted_path.is_file())
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["text"], source_text)

            with persisted_path.open(encoding="utf-8", newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))

        self.assertEqual(
            rows,
            [{"text": source_text, "chunk_size": str(len(source_text))}],
        )

    def test_cosine_similarity_distinguishes_alignment(self):
        aligned = self.agent.calculate_similarity([2.0, 0.0], [4.0, 0.0])
        orthogonal = self.agent.calculate_similarity([1.0, 0.0], [0.0, 3.0])

        self.assertAlmostEqual(aligned, 1.0)
        self.assertAlmostEqual(orthogonal, 0.0)


class RoutingAgentTests(unittest.TestCase):
    def test_route_selects_the_closest_specialist(self):
        product_handler = Mock(return_value="feature plan")
        operations_handler = Mock(return_value="incident response")
        routes = [
            {
                "name": "Product Planning",
                "description": "Group approved user stories into product features.",
                "func": product_handler,
            },
            {
                "name": "Operations",
                "description": "Coordinate incident response and service recovery.",
                "func": operations_handler,
            },
        ]
        router = RoutingAgent(
            "https://example.invalid/v1",
            "test-key",
            routes,
        )
        request = "Organize these user stories into a feature plan."
        embeddings = {
            request: [1.0, 0.0],
            routes[0]["description"]: [0.95, 0.05],
            routes[1]["description"]: [0.0, 1.0],
        }

        with patch.object(router, "get_embedding", side_effect=embeddings.__getitem__):
            with redirect_stdout(StringIO()):
                result = router.route(request)

        self.assertEqual(result, "feature plan")
        product_handler.assert_called_once_with(request)
        operations_handler.assert_not_called()


class EvaluationAgentTests(unittest.TestCase):
    def test_rejected_response_is_revised_and_then_accepted(self):
        worker = Mock()
        revised_response = (
            "Feature Name: Confidence-Aware Routing\n"
            "Description: Routes messages by intent and confidence."
        )
        worker.respond.side_effect = [
            "Feature Name: Routing",
            revised_response,
        ]
        client = Mock()
        client.chat.completions.create.side_effect = [
            completion_response("No. The description and grounding are incomplete."),
            completion_response(
                "Add a grounded description that explains intent and confidence "
                "routing."
            ),
            completion_response("Yes. The revised feature satisfies the criteria."),
        ]
        evaluator = EvaluationAgent(
            base_url="https://example.invalid/v1",
            openai_api_key="test-key",
            model_name="test-model",
            reasoning_effort="low",
            persona="Evaluate feature quality and specification grounding.",
            evaluation_criteria="Require a feature name and grounded description.",
            worker_agent=worker,
            max_interactions=3,
        )

        with patch(
            "workflow_agents.base_agents.OpenAI",
            return_value=client,
        ) as openai_client:
            with redirect_stdout(StringIO()):
                result = evaluator.evaluate(
                    "Define the confidence-aware routing feature."
                )

        self.assertEqual(result["final_response"], revised_response)
        self.assertTrue(result["evaluation"].startswith("Yes"))
        self.assertEqual(result["n_iterations"], 2)
        self.assertEqual(worker.respond.call_count, 2)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        openai_client.assert_called_once_with(
            base_url="https://example.invalid/v1",
            api_key="test-key",
        )

        revision_request = worker.respond.call_args_list[1].args[0]
        self.assertIn("Rejected response:", revision_request)
        self.assertIn("Correction instructions:", revision_request)
        self.assertIn("intent and confidence routing", revision_request)


if __name__ == "__main__":
    unittest.main()
