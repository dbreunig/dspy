"""
Example test file demonstrating how to use the interpreter conformance test suite.

This file tests the MockInterpreter against the conformance suite to verify that
the test suite itself works correctly and to serve as an example for community
implementers.

To run these tests:
    pytest tests/test_interpreter_conformance_example.py -v

Community implementers can use this as a template:
    1. Copy this file
    2. Replace MockInterpreter with your interpreter class
    3. Adjust fixtures for your implementation's capabilities
"""

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult
from tests.interpreter_conformance import InterpreterConformanceTests
from tests.mock_interpreter import MockInterpreter


class SimpleExecutingInterpreter:
    """A simple interpreter that actually executes Python code.

    This is used for conformance testing to validate the test suite.
    NOT a real sandboxed interpreter - for testing purposes only.
    """

    def __init__(self, tools=None, output_fields=None):
        self._tools = dict(tools or {})
        self.output_fields = output_fields or [{"name": "answer"}]
        self._namespace = {}
        self._started = False
        self._shutdown = False

    @property
    def tools(self):
        return self._tools

    def start(self):
        if self._started:
            return
        self._namespace = {"__builtins__": __builtins__}
        self._register_builtins()
        self._started = True
        self._shutdown = False

    def _register_builtins(self):
        self._output = []
        self._final_result = None

        def capture_print(*args, **kwargs):
            self._output.append(" ".join(str(a) for a in args))

        output_fields = self.output_fields

        def FINAL(*args, **kwargs):
            if args and not kwargs:
                # Positional args map to output field names in order
                for i, value in enumerate(args):
                    if i < len(output_fields):
                        kwargs[output_fields[i]["name"]] = value
                    else:
                        kwargs[f"arg_{i}"] = value
            self._final_result = FinalAnswerResult(kwargs)
            # Raise a special exception to stop execution
            raise _FinalCalled()

        def FINAL_VAR(*var_names):
            if len(var_names) != len(output_fields):
                raise ValueError(
                    f"FINAL_VAR expects {len(output_fields)} variable names "
                    f"for output fields {[f['name'] for f in output_fields]}, "
                    f"got {len(var_names)}"
                )
            result = {}
            for i, name in enumerate(var_names):
                if name not in self._namespace:
                    raise NameError(f"Variable '{name}' is not defined")
                field_name = output_fields[i]["name"] if i < len(output_fields) else name
                result[field_name] = self._namespace[name]
            self._final_result = FinalAnswerResult(result)
            raise _FinalCalled()

        self._namespace["print"] = capture_print
        self._namespace["FINAL"] = FINAL
        self._namespace["FINAL_VAR"] = FINAL_VAR

    def _register_tools(self):
        """Register tools in namespace."""
        for name, func in self._tools.items():
            self._namespace[name] = func

    def execute(self, code, variables=None):
        if self._shutdown:
            raise CodeInterpreterError("Interpreter has been shut down")

        if not self._started:
            self.start()

        # Reset output state
        self._output = []
        self._final_result = None

        # Inject variables
        if variables:
            self._namespace.update(variables)

        # Register/re-register tools (may have been updated)
        self._register_tools()

        # Handle empty/whitespace code
        if not code or not code.strip():
            return None

        # Check syntax
        try:
            compiled = compile(code, "<string>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax: {e}")

        # Execute
        try:
            exec(compiled, self._namespace)
        except _FinalCalled:
            return self._final_result
        except Exception as e:
            raise CodeInterpreterError(f"{type(e).__name__}: {e}")

        # Return result
        if self._final_result:
            return self._final_result
        if self._output:
            return "\n".join(self._output) + "\n"
        return None

    def shutdown(self):
        self._namespace = {}
        self._started = False
        self._shutdown = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()


class _FinalCalled(Exception):
    """Internal exception to signal FINAL was called."""

    pass


# =============================================================================
# Conformance Tests for SimpleExecutingInterpreter
# =============================================================================


class TestSimpleInterpreterConformance(InterpreterConformanceTests):
    """Run conformance tests against SimpleExecutingInterpreter."""

    @pytest.fixture
    def interpreter(self):
        """Provide interpreter instance."""
        return SimpleExecutingInterpreter()

    @pytest.fixture
    def skip_tool_tests(self):
        """Don't skip tool tests - we support them."""
        return False

    @pytest.fixture
    def skip_typed_final_tests(self):
        """Don't skip typed FINAL tests."""
        return False


# =============================================================================
# Additional Tests Specific to MockInterpreter
# =============================================================================


class TestMockInterpreterBasics:
    """Basic tests for MockInterpreter itself (not conformance)."""

    def test_mock_returns_scripted_responses(self):
        """MockInterpreter should return scripted responses in order."""
        mock = MockInterpreter(responses=["first", "second", "third"])
        mock.start()

        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

        mock.shutdown()

    def test_mock_returns_final_answer_result(self):
        """MockInterpreter should return FinalAnswerResult from responses."""
        final = FinalAnswerResult({"answer": "test"})
        mock = MockInterpreter(responses=[final])
        mock.start()

        result = mock.execute("some code")
        assert isinstance(result, FinalAnswerResult)
        assert result.answer == {"answer": "test"}

        mock.shutdown()

    def test_mock_raises_exceptions(self):
        """MockInterpreter should raise exceptions from responses."""
        error = CodeInterpreterError("test error")
        mock = MockInterpreter(responses=[error])
        mock.start()

        with pytest.raises(CodeInterpreterError, match="test error"):
            mock.execute("code")

        mock.shutdown()

    def test_mock_records_call_history(self):
        """MockInterpreter should record call history."""
        mock = MockInterpreter(responses=["response"])
        mock.start()

        mock.execute("test code", variables={"x": 1})

        assert len(mock.call_history) == 1
        assert mock.call_history[0] == ("test code", {"x": 1})

        mock.shutdown()

    def test_mock_with_custom_execute_fn(self):
        """MockInterpreter should use custom execute_fn if provided."""

        def custom_fn(code, variables):
            if "FINAL" in code:
                return FinalAnswerResult({"answer": "custom"})
            return f"executed: {code}"

        mock = MockInterpreter(execute_fn=custom_fn)
        mock.start()

        assert mock.execute("some code") == "executed: some code"
        result = mock.execute("FINAL()")
        assert isinstance(result, FinalAnswerResult)

        mock.shutdown()


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
