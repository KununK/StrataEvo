import json
import os
import unittest
from unittest.mock import patch

from tinyagent import Message, OpenAICompatibleModel, ToolCall, Usage
from tinyagent.model import DEFAULT_BASE_URL, DEFAULT_MODEL


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


class ModelTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_local_qwen_defaults(self):
        model = OpenAICompatibleModel()
        self.assertEqual(model.model, DEFAULT_MODEL)
        self.assertEqual(model.base_url, DEFAULT_BASE_URL)

    @patch("urllib.request.urlopen")
    def test_openai_compatible_model_parses_tool_call(self, urlopen):
        urlopen.return_value = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query":"agents"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        model = OpenAICompatibleModel(
            "test-model", api_key="secret", base_url="http://model.test/v1"
        )
        response = model.complete([Message("user", "find papers")], [])

        self.assertEqual(
            response.message.tool_calls, [ToolCall("call_1", "search", {"query": "agents"})]
        )
        self.assertEqual(response.usage, Usage(12, 4))
        request = urlopen.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["messages"][0]["content"], "find papers")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")

    @patch("urllib.request.urlopen")
    def test_structured_request_options_are_sent(self, urlopen):
        urlopen.return_value = FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        model = OpenAICompatibleModel("test-model", base_url="http://model.test/v1")

        model.complete(
            [Message("user", "return json")],
            [],
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["response_format"], {"type": "json_object"})

    @patch("urllib.request.urlopen")
    def test_empty_tool_result_is_sent_as_a_string(self, urlopen):
        urlopen.return_value = FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        model = OpenAICompatibleModel("test-model", base_url="http://model.test/v1")

        model.complete(
            [Message("tool", "", tool_call_id="call_1", name="read_file")],
            [],
        )

        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["messages"][0]["content"], "")

    @patch("urllib.request.urlopen", side_effect=TimeoutError)
    def test_timeout_has_a_clear_model_error(self, _urlopen):
        model = OpenAICompatibleModel(
            "test-model",
            base_url="http://model.test/v1",
            timeout=30,
        )

        with self.assertRaisesRegex(RuntimeError, "timed out after 30s"):
            model.complete([Message("user", "hello")], [])

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "http://environment.test/v1"})
    def test_explicit_base_url_has_priority_over_environment(self):
        model = OpenAICompatibleModel("test-model", base_url="http://explicit.test/v1")
        self.assertEqual(model.base_url, "http://explicit.test/v1")


if __name__ == "__main__":
    unittest.main()
