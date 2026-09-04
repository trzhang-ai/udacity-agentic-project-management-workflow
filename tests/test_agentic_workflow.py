"""Offline integration tests for the Email Router planning workflow."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock

from email_router.agentic_workflow import EmailRouterPlanningWorkflow
from email_router.agentic_workflow import WorkflowConfig


class EmailRouterPlanningWorkflowTests(unittest.TestCase):
    def test_pipeline_builds_all_artifacts_and_resets_state_between_runs(self):
        workflow = EmailRouterPlanningWorkflow(
            WorkflowConfig(
                base_url="https://example.invalid/v1",
                api_key="test-key",
                model_name="test-model",
                reasoning_effort="low",
            ),
            (
                "Classify incoming email, automate grounded routine responses, "
                "and route complex requests to the appropriate owner."
            ),
        )

        steps = [
            "Create user stories from the product specification",
            "Group those user stories into product features",
            "Create engineering tasks for those user stories and features",
        ]
        planner = Mock()
        planner.extract_steps_from_prompt.return_value = steps
        workflow.action_planning_agent = planner

        first_story = (
            "As a service owner, I want intent classification so that email reaches "
            "the correct handler."
        )
        second_story = (
            "As a support lead, I want confidence-aware routing so that ambiguous "
            "email receives human review."
        )
        first_feature = (
            "Feature Name: Intent Routing\n"
            "Description: Routes email by detected intent.\n"
            "Key Functionality: Intent classification\n"
            "User Benefit: Faster ownership"
        )
        second_feature = (
            "Feature Name: Confidence-Aware Triage\n"
            "Description: Escalates ambiguous email for human review.\n"
            "Key Functionality: Confidence thresholds\n"
            "User Benefit: Safer automation"
        )
        first_task = (
            "Task ID: ROUTE-1\n"
            "Task Title: Implement intent routing\n"
            "Related User Story: service owner intent classification\n"
            "Description: Build routing logic.\n"
            "Acceptance Criteria: Correct owner is selected.\n"
            "Estimated Effort: Medium\n"
            "Dependencies: None"
        )
        second_task = (
            "Task ID: TRIAGE-1\n"
            "Task Title: Implement confidence thresholds\n"
            "Related User Story: support lead confidence-aware routing\n"
            "Description: Escalate ambiguous classifications.\n"
            "Acceptance Criteria: Low-confidence email enters human review.\n"
            "Estimated Effort: Medium\n"
            "Dependencies: Intent classifier"
        )

        product_evaluator = Mock()
        product_evaluator.evaluate.side_effect = [
            {"final_response": first_story},
            {"final_response": second_story},
        ]
        program_evaluator = Mock()
        program_evaluator.evaluate.side_effect = [
            {"final_response": first_feature},
            {"final_response": second_feature},
        ]
        engineering_evaluator = Mock()
        engineering_evaluator.evaluate.side_effect = [
            {"final_response": first_task},
            {"final_response": second_task},
        ]
        workflow.product_manager_evaluator = product_evaluator
        workflow.program_manager_evaluator = program_evaluator
        workflow.development_engineer_evaluator = engineering_evaluator

        route_handlers = {
            steps[0]: workflow.product_manager_support_function,
            steps[1]: workflow.program_manager_support_function,
            steps[2]: workflow.development_engineer_support_function,
        }
        router = Mock()
        router.route.side_effect = lambda step: route_handlers[step](step)
        workflow.routing_agent = router

        with redirect_stdout(StringIO()):
            first_output = workflow.run("Build the Email Router delivery plan.")
            second_output = workflow.run("Refresh the Email Router delivery plan.")

        self.assertIn(first_story, first_output)
        self.assertIn(first_feature, first_output)
        self.assertIn(first_task, first_output)

        self.assertNotIn(first_story, second_output)
        self.assertNotIn(first_feature, second_output)
        self.assertNotIn(first_task, second_output)
        self.assertIn(second_story, second_output)
        self.assertIn(second_feature, second_output)
        self.assertIn(second_task, second_output)
        self.assertIn("USER STORIES", second_output)
        self.assertIn("PRODUCT FEATURES", second_output)
        self.assertIn("ENGINEERING TASKS", second_output)

        self.assertEqual(
            workflow.workflow_state,
            {
                "user_stories": [second_story],
                "features": [second_feature],
                "engineering_tasks": [second_task],
            },
        )
        self.assertEqual(planner.extract_steps_from_prompt.call_count, 2)
        self.assertEqual(router.route.call_count, 6)
        self.assertEqual(product_evaluator.evaluate.call_count, 2)
        self.assertEqual(program_evaluator.evaluate.call_count, 2)
        self.assertEqual(engineering_evaluator.evaluate.call_count, 2)

        second_feature_request = program_evaluator.evaluate.call_args_list[1].args[0]
        second_task_request = engineering_evaluator.evaluate.call_args_list[1].args[0]
        self.assertIn(second_story, second_feature_request)
        self.assertIn(second_story, second_task_request)
        self.assertIn(second_feature, second_task_request)


if __name__ == "__main__":
    unittest.main()
