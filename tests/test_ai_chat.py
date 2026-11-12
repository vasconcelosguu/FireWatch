import unittest

from ai.chat_assistant import answer_firewatch_question
from ai.report_generator import ReportGenerationError, prepare_report_data


class FakeCompletions:
    def __init__(self, content="Há 2 registros no histórico.", error=None):
        self.content = content
        self.error = error
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        if self.error:
            raise self.error
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


class ChatAssistantTests(unittest.TestCase):
    def setUp(self):
        self.context = prepare_report_data([
            {"timestamp": "2026-11-09 14:30:00", "source": "video", "class_name": "Fire", "confidence": 0.9, "image_path": "C:/private/fire.jpg", "x1": 10},
            {"timestamp": "2026-11-09 14:31:00", "source": "webcam", "class_name": "Smoke", "confidence": 0.7},
        ])

    def test_answers_using_history_and_question(self):
        completions = FakeCompletions("O histórico contém registros Fire e Smoke.")
        answer = answer_firewatch_question(
            "Quais classes aparecem?", self.context,
            history=[{"role": "user", "content": "Oi"}],
            client=FakeClient(completions), model="openai/gpt-oss-20b",
        )
        self.assertIn("Fire e Smoke", answer)
        request = completions.last_request
        self.assertEqual(request["model"], "openai/gpt-oss-20b")
        self.assertEqual(request["messages"][1]["content"], "Oi")
        final_prompt = request["messages"][-1]["content"]
        self.assertIn("Quais classes aparecem?", final_prompt)
        self.assertNotIn("C:/private", final_prompt)
        self.assertNotIn("fire.jpg", final_prompt)
        self.assertNotIn("x1", final_prompt)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            answer_firewatch_question("  ", self.context, client=FakeClient(FakeCompletions()))

    def test_empty_context_is_rejected(self):
        with self.assertRaises(ValueError):
            answer_firewatch_question("Quantas detecções?", {}, client=FakeClient(FakeCompletions()))

    def test_api_failure_is_wrapped(self):
        with self.assertRaises(ReportGenerationError):
            answer_firewatch_question(
                "Resumo?", self.context,
                client=FakeClient(FakeCompletions(error=RuntimeError("internal"))),
            )


if __name__ == "__main__":
    unittest.main()
