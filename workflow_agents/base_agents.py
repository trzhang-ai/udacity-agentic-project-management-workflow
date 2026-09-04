"""Reusable agent components for prompt orchestration and semantic routing."""

from __future__ import annotations

import ast
import csv
from datetime import datetime
from pathlib import Path
import re
import uuid

import numpy as np
from openai import OpenAI
import pandas as pd


class DirectPromptAgent:
    """Send a user prompt to a chat-completion model without added context."""

    def __init__(self, openai_api_key, base_url, model_name):
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.model_name = model_name

    def respond(self, user_prompt):
        """Return the model's text response."""
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content.strip()


class AugmentedPromptAgent:
    """Apply a role and response style to an otherwise direct prompt."""

    def __init__(self, openai_api_key, base_url, persona, model_name):
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.persona = persona
        self.model_name = model_name

    def respond(self, input_text):
        """Return a response shaped by the configured persona."""
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        f"{self.persona}\n"
                        "Treat this request as self-contained. Use only context "
                        "provided in the current request."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
        )
        return response.choices[0].message.content.strip()


class KnowledgeAugmentedPromptAgent:
    """Generate answers from bounded knowledge and explicit request context."""

    def __init__(
        self, base_url, openai_api_key, persona, knowledge, model_name, reasoning_effort
    ):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def respond(self, input_text):
        """Return a knowledge-grounded response or evaluator-directed revision."""
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "developer",
                    "content": f"""
                        Role and default style:
                        {self.persona}

                        Treat this request as self-contained. Use all product
                        specifications, workflow artifacts, and evaluator feedback
                        explicitly included in the current request. Do not use other
                        conversational history.

                        <external_knowledge>
                        {self.knowledge}
                        </external_knowledge>

                        Operate in one of two modes:

                        1. Initial-answer mode
                        - For a request without evaluator feedback, use
                          <external_knowledge> together with specifications and workflow
                          artifacts included in the request.
                        - Do not add unsupported factual knowledge.
                        - Follow the default role and style.

                        2. Evaluator-revision mode
                        - When the request includes a rejected response and correction
                          instructions, apply those instructions exactly.
                        - Corrections may change facts, wording, formatting, and style.
                        - Evaluator corrections override <external_knowledge>, the default
                          role, and the rejected response.
                        - Return only the corrected answer. Do not discuss the revision
                          process or defend the previous answer.
                        """.strip(),
                },
                {"role": "user", "content": input_text},
            ],
            reasoning_effort=self.reasoning_effort,
        )
        return response.choices[0].message.content.strip()


class RAGKnowledgePromptAgent:
    """Retrieve the most relevant text chunk before generating a grounded answer."""

    def __init__(
        self,
        base_url,
        openai_api_key,
        persona,
        model_name,
        reasoning_effort,
        chunk_size=2000,
        chunk_overlap=100,
    ):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    @property
    def chunks_path(self):
        """Return the generated chunk-store path for this agent instance."""
        return Path(f"chunks-{self.unique_filename}")

    @property
    def embeddings_path(self):
        """Return the generated embedding-store path for this agent instance."""
        return Path(f"embeddings-{self.unique_filename}")

    def get_embedding(self, text):
        """Return an embedding vector for non-empty text."""
        if text is None or not text.strip():
            raise ValueError("text must be a non-empty string")

        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    @staticmethod
    def calculate_similarity(vector_one, vector_two):
        """Calculate cosine similarity between two embedding vectors."""
        vec1 = np.asarray(vector_one, dtype=float)
        vec2 = np.asarray(vector_two, dtype=float)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            raise ValueError("embedding vectors must have non-zero magnitude")
        return float(np.dot(vec1, vec2) / denominator)

    def chunk_text(self, text):
        """Split text into overlapping chunks and persist the chunk store."""
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if text is None or not text.strip():
            raise ValueError("text must be a non-empty string")

        normalized_text = re.sub(r"[ \t]+", " ", text)
        normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()
        chunks = []
        start = 0

        while start < len(normalized_text):
            end = min(start + self.chunk_size, len(normalized_text))

            if end < len(normalized_text):
                newline_break = normalized_text.rfind("\n", start, end)
                sentence_break = normalized_text.rfind(". ", start, end)
                natural_break = max(newline_break, sentence_break)
                if natural_break > start + self.chunk_overlap:
                    end = natural_break + (2 if natural_break == sentence_break else 1)

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": normalized_text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            if end >= len(normalized_text):
                break
            start = end - self.chunk_overlap

        with self.chunks_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow(
                    {"text": chunk["text"], "chunk_size": chunk["chunk_size"]}
                )

        return chunks

    def calculate_embeddings(self):
        """Embed the persisted chunks and write an embedding store."""
        dataframe = pd.read_csv(self.chunks_path, encoding="utf-8")
        dataframe["embeddings"] = dataframe["text"].apply(self.get_embedding)
        dataframe.to_csv(self.embeddings_path, encoding="utf-8", index=False)
        return dataframe

    def find_prompt_in_knowledge(self, prompt):
        """Retrieve the closest chunk and answer only from that context."""
        prompt_embedding = self.get_embedding(prompt)
        dataframe = pd.read_csv(self.embeddings_path, encoding="utf-8")
        dataframe["embeddings"] = dataframe["embeddings"].apply(
            lambda value: np.asarray(ast.literal_eval(value), dtype=float)
        )
        dataframe["similarity"] = dataframe["embeddings"].apply(
            lambda embedding: self.calculate_similarity(prompt_embedding, embedding)
        )
        best_chunk = dataframe.loc[dataframe["similarity"].idxmax(), "text"]

        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        f"You are {self.persona}. Treat this request as self-contained "
                        "and answer only from the retrieved context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Retrieved context:\n{best_chunk}\n\n"
                        f"Request:\n{prompt}"
                    ),
                },
            ],
            reasoning_effort=self.reasoning_effort,
        )
        return response.choices[0].message.content.strip()


class EvaluationAgent:
    """Revise a worker response until it satisfies explicit acceptance criteria."""

    def __init__(
        self,
        base_url,
        openai_api_key,
        model_name,
        reasoning_effort,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
    ):
        if max_interactions < 1:
            raise ValueError("max_interactions must be at least 1")

        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.persona = persona
        self.reasoning_effort = reasoning_effort
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """Return the first accepted worker response and its evaluation metadata."""
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for iteration in range(1, self.max_interactions + 1):
            print(f"\n--- Evaluation cycle {iteration} ---")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker response:\n{response_from_worker}")

            evaluation_prompt = (
                f"Original task and supplied context:\n{initial_prompt}\n\n"
                f"Candidate answer:\n{response_from_worker}\n\n"
                f"Evaluation criteria:\n{self.evaluation_criteria}\n\n"
                "Evaluate both the required format and whether the candidate answer "
                "is grounded in the original task and supplied context. Begin with "
                "the plain-text word Yes or No, without Markdown, then explain why."
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "developer", "content": self.persona},
                    {"role": "user", "content": evaluation_prompt},
                ],
                reasoning_effort=self.reasoning_effort,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator verdict:\n{evaluation}")

            normalized_evaluation = evaluation.strip().lstrip("*_`# ")
            if normalized_evaluation.lower().startswith("yes"):
                print("Response accepted.")
                return {
                    "final_response": response_from_worker,
                    "evaluation": evaluation,
                    "n_iterations": iteration,
                }

            instruction_prompt = (
                f"Original task and supplied context:\n{initial_prompt}\n\n"
                f"Candidate answer:\n{response_from_worker}\n\n"
                f"Evaluator feedback:\n{evaluation}\n\n"
                "Provide precise correction instructions. The revision must remain "
                "grounded in the original task and supplied context and must not "
                "introduce an unrelated product or placeholder example."
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "developer", "content": self.persona},
                    {"role": "user", "content": instruction_prompt},
                ],
                reasoning_effort=self.reasoning_effort,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Revision instructions:\n{instructions}")

            prompt_to_evaluate = (
                f"Original request:\n{initial_prompt}\n\n"
                f"Rejected response:\n{response_from_worker}\n\n"
                f"Correction instructions:\n{instructions}\n\n"
                "Return only the corrected response."
            )

        raise RuntimeError(
            "The worker response did not satisfy the evaluation criteria "
            f"after {self.max_interactions} interactions. "
            f"Last evaluation: {evaluation}"
        )


class RoutingAgent:
    """Select a specialist by comparing semantic similarity to route descriptions."""

    def __init__(self, base_url, openai_api_key, agents):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """Return an embedding vector, or ``None`` for empty input."""
        if text is None or not text.strip():
            return None

        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
        )
        return response.data[0].embedding

    def route(self, user_input):
        """Dispatch the input to the closest matching configured agent."""
        input_embedding = self.get_embedding(user_input)
        if input_embedding is None:
            return "No suitable agent could be selected for an empty request."

        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            agent_embedding = self.get_embedding(agent["description"])
            if agent_embedding is None:
                continue

            similarity = RAGKnowledgePromptAgent.calculate_similarity(
                input_embedding, agent_embedding
            )
            if similarity > best_score:
                best_agent = agent
                best_score = similarity

        if best_agent is None:
            return "No suitable agent could be selected."

        print(f"[Router] Selected {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """Extract an ordered action plan constrained by a supplied playbook."""

    def __init__(
        self, base_url, openai_api_key, knowledge, model_name, reasoning_effort
    ):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def extract_steps_from_prompt(self, prompt):
        """Return newline-delimited plan steps as a list."""
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "developer",
                    "content": f"""
                    You are an action-planning agent. Convert the request into the
                    relevant ordered steps from the supplied playbook. Return only
                    those steps, one per line, without commentary. Treat each call as
                    self-contained.

                    <planning_playbook>
                    {self.knowledge}
                    </planning_playbook>
                    """.strip(),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=self.reasoning_effort,
        )
        response_text = response.choices[0].message.content.strip()
        return [step.strip() for step in response_text.splitlines() if step.strip()]
