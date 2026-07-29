import unittest
from typing import Literal

from tinyagent import ToolRegistry, tool


class ToolTests(unittest.TestCase):
    def test_tool_derives_json_schema_and_defaults(self):
        @tool(description="Search a corpus")
        def search(
            query: str, limit: int = 5, mode: Literal["exact", "fuzzy"] = "exact"
        ) -> list[str]:
            return [query] * limit

        function = search.schema["function"]
        self.assertEqual(function["name"], "search")
        self.assertEqual(function["description"], "Search a corpus")
        self.assertEqual(function["parameters"]["required"], ["query"])
        self.assertEqual(
            function["parameters"]["properties"]["limit"], {"type": "integer", "default": 5}
        )
        self.assertEqual(function["parameters"]["properties"]["mode"]["enum"], ["exact", "fuzzy"])

    def test_registry_rejects_duplicates(self):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        registry = ToolRegistry([ping])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register(ping)

    def test_registry_can_append_descriptions_without_mutating_tools(self):
        @tool(description="Read one file")
        def read(path: str) -> str:
            return path

        original = ToolRegistry([read])
        evolved = original.with_description_addenda({"read": "Inspect relevant files first."})

        self.assertEqual(original.schemas[0]["function"]["description"], "Read one file")
        self.assertEqual(
            evolved.schemas[0]["function"]["description"],
            "Read one file\n\nInspect relevant files first.",
        )
        self.assertIs(evolved.get("read").function, read.function)
        self.assertEqual(evolved.get("read").requires_approval, read.requires_approval)

    def test_registry_rejects_unknown_description_addenda(self):
        with self.assertRaisesRegex(ValueError, "unknown tools"):
            ToolRegistry().with_description_addenda({"missing": "guidance"})

    def test_optional_uses_standard_json_schema(self):
        @tool
        def lookup(year: int | None = None) -> str:
            """Look up a year."""
            return str(year)

        year = lookup.parameters["properties"]["year"]
        self.assertEqual(year["anyOf"], [{"type": "integer"}, {"type": "null"}])

    def test_function_type_error_is_not_reported_as_bad_arguments(self):
        @tool
        def broken(value: str) -> str:
            """Raise an internal error."""
            raise TypeError("inside function")

        with self.assertRaisesRegex(TypeError, "inside function"):
            broken.run({"value": "x"})

        with self.assertRaisesRegex(ValueError, "invalid arguments"):
            broken.run({})


if __name__ == "__main__":
    unittest.main()
