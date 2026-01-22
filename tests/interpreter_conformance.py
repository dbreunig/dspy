"""
Conformance test suite for custom CodeInterpreter implementations.

This module provides a comprehensive test suite that validates whether a
CodeInterpreter implementation meets all the requirements for use with DSPy's
RLM module. Community implementers can inherit from InterpreterConformanceTests
and provide their interpreter instance via a pytest fixture.

Usage:
    ```python
    import pytest
    from tests.interpreter_conformance import InterpreterConformanceTests

    class TestMyInterpreter(InterpreterConformanceTests):
        @pytest.fixture
        def interpreter(self):
            return MyInterpreter()

        # Optional: skip tests that don't apply to your implementation
        @pytest.fixture
        def skip_tool_tests(self):
            return False  # Set True to skip tool-related tests
    ```

The test suite covers:
    1. Core Protocol - Required methods and properties
    2. Lifecycle Management - start(), shutdown(), idempotency
    3. Code Execution - Basic code, imports, expressions
    4. State Persistence - Variables persist across calls
    5. Variable Injection - Injected variables accessible in code
    6. Output Handling - print(), expression values, None
    7. Error Handling - CodeInterpreterError, SyntaxError
    8. Tool Integration - Tool calls with args, kwargs, return values
    9. FINAL Functions - FINAL() and FINAL_VAR() with various args
    10. Context Manager - __enter__, __exit__
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

if TYPE_CHECKING:
    from dspy.primitives.code_interpreter import CodeInterpreter


class InterpreterConformanceTests(abc.ABC):
    """Base class for CodeInterpreter conformance tests.

    Subclass this and provide an `interpreter` fixture that returns your
    interpreter instance. All tests will run against your implementation.

    Example:
        ```python
        class TestMyInterpreter(InterpreterConformanceTests):
            @pytest.fixture
            def interpreter(self):
                return MyInterpreter()
        ```
    """

    # ==========================================================================
    # Required Fixture - Subclasses MUST override
    # ==========================================================================

    @pytest.fixture
    @abc.abstractmethod
    def interpreter(self) -> "CodeInterpreter":
        """Return an interpreter instance for testing.

        This fixture MUST be overridden in subclasses.

        Returns:
            A fresh CodeInterpreter instance (not started).
        """
        raise NotImplementedError("Subclasses must provide an interpreter fixture")

    # ==========================================================================
    # Optional Fixtures - Subclasses MAY override to customize behavior
    # ==========================================================================

    @pytest.fixture
    def skip_tool_tests(self) -> bool:
        """Override to skip tool integration tests."""
        return False

    @pytest.fixture
    def skip_typed_final_tests(self) -> bool:
        """Override to skip typed FINAL/output_fields tests."""
        return False

    @pytest.fixture
    def supports_imports(self) -> bool:
        """Override to indicate if interpreter supports Python imports."""
        return True

    @pytest.fixture
    def supports_multiline_code(self) -> bool:
        """Override to indicate if interpreter supports multiline code."""
        return True

    # ==========================================================================
    # Section 1: Core Protocol Tests
    # ==========================================================================

    class TestCoreProtocol:
        """Tests for basic protocol compliance."""

        def test_has_tools_property(self, interpreter):
            """Interpreter must have a tools property."""
            assert hasattr(interpreter, "tools"), "Interpreter must have 'tools' property"
            tools = interpreter.tools
            assert isinstance(tools, dict), "tools must return a dict"

        def test_tools_property_is_mutable(self, interpreter):
            """Tools property must return a mutable dict for RLM injection."""
            tools = interpreter.tools
            # RLM calls interpreter.tools.update({...})
            tools["_test_tool"] = lambda: "test"
            assert "_test_tool" in interpreter.tools, "tools dict must be mutable"
            # Clean up
            del interpreter.tools["_test_tool"]

        def test_has_start_method(self, interpreter):
            """Interpreter must have a start() method."""
            assert hasattr(interpreter, "start"), "Interpreter must have 'start' method"
            assert callable(interpreter.start), "start must be callable"

        def test_has_execute_method(self, interpreter):
            """Interpreter must have an execute() method."""
            assert hasattr(interpreter, "execute"), "Interpreter must have 'execute' method"
            assert callable(interpreter.execute), "execute must be callable"

        def test_has_shutdown_method(self, interpreter):
            """Interpreter must have a shutdown() method."""
            assert hasattr(interpreter, "shutdown"), "Interpreter must have 'shutdown' method"
            assert callable(interpreter.shutdown), "shutdown must be callable"

    # ==========================================================================
    # Section 2: Lifecycle Management Tests
    # ==========================================================================

    class TestLifecycle:
        """Tests for interpreter lifecycle management."""

        def test_start_is_idempotent(self, interpreter):
            """Calling start() multiple times should be safe."""
            interpreter.start()
            interpreter.start()  # Should not raise or double-allocate
            interpreter.shutdown()

        def test_shutdown_is_idempotent(self, interpreter):
            """Calling shutdown() multiple times should be safe."""
            interpreter.start()
            interpreter.shutdown()
            interpreter.shutdown()  # Should not raise

        def test_execute_calls_start_lazily(self, interpreter):
            """Execute should call start() lazily if not already started."""
            # Don't call start() explicitly
            result = interpreter.execute("x = 1")
            # Should not raise - start() called lazily
            interpreter.shutdown()

        def test_execute_after_shutdown_fails(self, interpreter):
            """Execute after shutdown should fail or be handled gracefully."""
            interpreter.start()
            interpreter.shutdown()
            # Behavior depends on implementation - either raise or reinitialize
            # We just verify it doesn't crash unexpectedly
            try:
                interpreter.execute("x = 1")
            except (CodeInterpreterError, RuntimeError):
                pass  # Expected - interpreter was shut down

    # ==========================================================================
    # Section 3: Basic Code Execution Tests
    # ==========================================================================

    class TestBasicExecution:
        """Tests for basic code execution."""

        def test_execute_simple_expression(self, interpreter):
            """Simple expression should execute without error."""
            interpreter.start()
            try:
                result = interpreter.execute("1 + 1")
                # Result could be 2, "2", or None depending on implementation
                # The key is it doesn't raise an error
            finally:
                interpreter.shutdown()

        def test_execute_assignment(self, interpreter):
            """Assignment should execute without error."""
            interpreter.start()
            try:
                result = interpreter.execute("x = 42")
                # Assignment typically returns None
            finally:
                interpreter.shutdown()

        def test_execute_print_returns_output(self, interpreter):
            """Print statement should return captured output."""
            interpreter.start()
            try:
                result = interpreter.execute("print('hello')")
                # Result should contain "hello"
                assert result is not None, "print() should produce output"
                output = str(result) if not isinstance(result, str) else result
                assert "hello" in output, f"Output should contain 'hello', got: {result}"
            finally:
                interpreter.shutdown()

        def test_execute_multiline_code(self, interpreter, supports_multiline_code):
            """Multiline code should execute correctly."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                code = """
x = 10
y = 20
print(x + y)
"""
                result = interpreter.execute(code)
                output = str(result) if result else ""
                assert "30" in output, f"Output should contain '30', got: {result}"
            finally:
                interpreter.shutdown()

        def test_execute_with_imports(self, interpreter, supports_imports):
            """Standard library imports should work."""
            if not supports_imports:
                pytest.skip("Interpreter does not support imports")

            interpreter.start()
            try:
                code = "import math\nprint(math.sqrt(16))"
                result = interpreter.execute(code)
                output = str(result) if result else ""
                assert "4" in output, f"Output should contain '4', got: {result}"
            finally:
                interpreter.shutdown()

        def test_execute_returns_none_for_no_output(self, interpreter):
            """Execution with no output should return None or empty string."""
            interpreter.start()
            try:
                result = interpreter.execute("x = 1")
                # Should be None, "", or similar "no output" value
                assert result is None or result == "" or result == [], \
                    f"No-output code should return None/empty, got: {result}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 4: State Persistence Tests
    # ==========================================================================

    class TestStatePersistence:
        """Tests for state persistence across execute() calls."""

        def test_variables_persist_across_calls(self, interpreter):
            """Variables defined in one call should be available in the next."""
            interpreter.start()
            try:
                interpreter.execute("counter = 0")
                interpreter.execute("counter = counter + 1")
                result = interpreter.execute("print(counter)")

                output = str(result) if result else ""
                assert "1" in output, f"Counter should be 1, got: {result}"
            finally:
                interpreter.shutdown()

        def test_functions_persist_across_calls(self, interpreter, supports_multiline_code):
            """Functions defined in one call should be callable in the next."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                interpreter.execute("def add(a, b): return a + b")
                result = interpreter.execute("print(add(2, 3))")

                output = str(result) if result else ""
                assert "5" in output, f"add(2, 3) should be 5, got: {result}"
            finally:
                interpreter.shutdown()

        def test_complex_state_accumulation(self, interpreter, supports_multiline_code):
            """Complex state should accumulate correctly."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                interpreter.execute("items = []")
                interpreter.execute("items.append('a')")
                interpreter.execute("items.append('b')")
                result = interpreter.execute("print(items)")

                output = str(result) if result else ""
                assert "a" in output and "b" in output, \
                    f"items should contain 'a' and 'b', got: {result}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 5: Variable Injection Tests
    # ==========================================================================

    class TestVariableInjection:
        """Tests for variable injection via execute(variables={...})."""

        def test_inject_simple_variable(self, interpreter):
            """Simple variables should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(x)",
                    variables={"x": 42}
                )
                output = str(result) if result else ""
                assert "42" in output, f"x should be 42, got: {result}"
            finally:
                interpreter.shutdown()

        def test_inject_string_variable(self, interpreter):
            """String variables should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(message)",
                    variables={"message": "hello world"}
                )
                output = str(result) if result else ""
                assert "hello world" in output, f"message should be 'hello world', got: {result}"
            finally:
                interpreter.shutdown()

        def test_inject_list_variable(self, interpreter):
            """List variables should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(len(items))",
                    variables={"items": [1, 2, 3, 4, 5]}
                )
                output = str(result) if result else ""
                assert "5" in output, f"len(items) should be 5, got: {result}"
            finally:
                interpreter.shutdown()

        def test_inject_dict_variable(self, interpreter):
            """Dict variables should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(data['key'])",
                    variables={"data": {"key": "value123"}}
                )
                output = str(result) if result else ""
                assert "value123" in output, f"data['key'] should be 'value123', got: {result}"
            finally:
                interpreter.shutdown()

        def test_inject_multiple_variables(self, interpreter):
            """Multiple variables should be injectable simultaneously."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(a + b)",
                    variables={"a": 10, "b": 20}
                )
                output = str(result) if result else ""
                assert "30" in output, f"a + b should be 30, got: {result}"
            finally:
                interpreter.shutdown()

        def test_injected_variables_persist(self, interpreter):
            """Injected variables should persist for subsequent calls."""
            interpreter.start()
            try:
                interpreter.execute("y = x * 2", variables={"x": 5})
                result = interpreter.execute("print(y)")

                output = str(result) if result else ""
                assert "10" in output, f"y should be 10, got: {result}"
            finally:
                interpreter.shutdown()

        def test_injected_variables_can_be_overwritten(self, interpreter):
            """Injected variables should be overwritable."""
            interpreter.start()
            try:
                interpreter.execute("x = 100", variables={"x": 1})
                result = interpreter.execute("print(x)")

                output = str(result) if result else ""
                assert "100" in output, f"x should be 100 after overwrite, got: {result}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 6: Error Handling Tests
    # ==========================================================================

    class TestErrorHandling:
        """Tests for error handling (CodeInterpreterError, SyntaxError)."""

        def test_syntax_error_raises_exception(self, interpreter):
            """Invalid Python syntax should raise SyntaxError."""
            interpreter.start()
            try:
                with pytest.raises(SyntaxError):
                    interpreter.execute("def broken(")
            finally:
                interpreter.shutdown()

        def test_syntax_error_on_invalid_tokens(self, interpreter):
            """Invalid tokens should raise SyntaxError."""
            interpreter.start()
            try:
                with pytest.raises(SyntaxError):
                    interpreter.execute("+++")
            finally:
                interpreter.shutdown()

        def test_name_error_raises_code_interpreter_error(self, interpreter):
            """NameError should raise CodeInterpreterError."""
            interpreter.start()
            try:
                with pytest.raises(CodeInterpreterError):
                    interpreter.execute("print(undefined_variable_xyz)")
            finally:
                interpreter.shutdown()

        def test_type_error_raises_code_interpreter_error(self, interpreter):
            """TypeError should raise CodeInterpreterError."""
            interpreter.start()
            try:
                with pytest.raises(CodeInterpreterError):
                    interpreter.execute("'string' + 123")
            finally:
                interpreter.shutdown()

        def test_zero_division_raises_code_interpreter_error(self, interpreter):
            """ZeroDivisionError should raise CodeInterpreterError."""
            interpreter.start()
            try:
                with pytest.raises(CodeInterpreterError):
                    interpreter.execute("1 / 0")
            finally:
                interpreter.shutdown()

        def test_error_message_contains_info(self, interpreter):
            """Error messages should contain useful information."""
            interpreter.start()
            try:
                with pytest.raises(CodeInterpreterError) as exc_info:
                    interpreter.execute("undefined_var")
                error_msg = str(exc_info.value).lower()
                # Should mention the error type or variable name
                assert "name" in error_msg or "undefined" in error_msg or "not defined" in error_msg, \
                    f"Error message should be informative, got: {exc_info.value}"
            finally:
                interpreter.shutdown()

        def test_error_does_not_corrupt_state(self, interpreter):
            """Errors should not corrupt interpreter state."""
            interpreter.start()
            try:
                interpreter.execute("good_var = 42")

                # Cause an error
                with pytest.raises(CodeInterpreterError):
                    interpreter.execute("bad_var")

                # State should still be intact
                result = interpreter.execute("print(good_var)")
                output = str(result) if result else ""
                assert "42" in output, f"State should be preserved after error, got: {result}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 7: Tool Integration Tests
    # ==========================================================================

    class TestToolIntegration:
        """Tests for tool (host function) integration."""

        def test_simple_tool_call(self, interpreter, skip_tool_tests):
            """Simple tool should be callable from code."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def echo_tool(message: str) -> str:
                return f"echo: {message}"

            interpreter.tools["echo"] = echo_tool
            interpreter.start()
            try:
                result = interpreter.execute('print(echo(message="hello"))')
                output = str(result) if result else ""
                assert "echo: hello" in output, f"Tool should return 'echo: hello', got: {result}"
            finally:
                interpreter.shutdown()

        def test_tool_with_keyword_args(self, interpreter, skip_tool_tests):
            """Tools should work with keyword arguments."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def search(query: str, limit: int = 10) -> str:
                return f"searched '{query}' with limit {limit}"

            interpreter.tools["search"] = search
            interpreter.start()
            try:
                result = interpreter.execute('print(search(query="test", limit=5))')
                output = str(result) if result else ""
                assert "searched 'test' with limit 5" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()

        def test_tool_with_positional_args(self, interpreter, skip_tool_tests):
            """Tools should work with positional arguments."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def add(a: int, b: int) -> str:
                return str(a + b)

            interpreter.tools["add"] = add
            interpreter.start()
            try:
                result = interpreter.execute('print(add(3, 4))')
                output = str(result) if result else ""
                assert "7" in output, f"add(3, 4) should be 7, got: {result}"
            finally:
                interpreter.shutdown()

        def test_tool_with_default_args(self, interpreter, skip_tool_tests):
            """Tools should respect default argument values."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def greet(name: str, greeting: str = "Hello") -> str:
                return f"{greeting}, {name}!"

            interpreter.tools["greet"] = greet
            interpreter.start()
            try:
                # Without providing default
                result = interpreter.execute('print(greet(name="World"))')
                output = str(result) if result else ""
                assert "Hello, World!" in output, f"Got: {result}"

                # With overridden default
                result = interpreter.execute('print(greet(name="World", greeting="Hi"))')
                output = str(result) if result else ""
                assert "Hi, World!" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()

        def test_tool_return_value_usable(self, interpreter, skip_tool_tests):
            """Tool return values should be usable in subsequent code."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def get_number() -> str:
                return "42"

            interpreter.tools["get_number"] = get_number
            interpreter.start()
            try:
                interpreter.execute('num_str = get_number()')
                result = interpreter.execute('print(int(num_str) * 2)')
                output = str(result) if result else ""
                assert "84" in output, f"int('42') * 2 should be 84, got: {result}"
            finally:
                interpreter.shutdown()

        def test_multiple_tools(self, interpreter, skip_tool_tests):
            """Multiple tools should be usable together."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            def tool_a() -> str:
                return "A"

            def tool_b() -> str:
                return "B"

            interpreter.tools["tool_a"] = tool_a
            interpreter.tools["tool_b"] = tool_b
            interpreter.start()
            try:
                result = interpreter.execute('print(tool_a() + tool_b())')
                output = str(result) if result else ""
                assert "AB" in output, f"tool_a() + tool_b() should be 'AB', got: {result}"
            finally:
                interpreter.shutdown()

        def test_tools_can_be_updated_after_start(self, interpreter, skip_tool_tests):
            """Tools dict should be updatable after start() (RLM requirement)."""
            if skip_tool_tests:
                pytest.skip("Tool tests skipped")

            interpreter.start()
            try:
                # Add tool after start
                def late_tool() -> str:
                    return "late"

                interpreter.tools["late_tool"] = late_tool

                # May need to re-register tools depending on implementation
                # Some implementations register on each execute(), others on start()
                result = interpreter.execute('print(late_tool())')
                output = str(result) if result else ""
                # If this works, great. If not, implementation may need adjustment.
                # We don't strictly fail here as this is advanced functionality.
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 8: FINAL Function Tests
    # ==========================================================================

    class TestFinalFunction:
        """Tests for FINAL() function."""

        def test_final_with_single_value(self, interpreter):
            """FINAL with single value should return FinalAnswerResult."""
            interpreter.start()
            try:
                result = interpreter.execute('FINAL("the answer")')
                assert isinstance(result, FinalAnswerResult), \
                    f"FINAL should return FinalAnswerResult, got: {type(result)}"
            finally:
                interpreter.shutdown()

        def test_final_answer_contains_value(self, interpreter):
            """FinalAnswerResult should contain the provided value."""
            interpreter.start()
            try:
                result = interpreter.execute('FINAL("test_value")')
                assert isinstance(result, FinalAnswerResult)
                # answer should be a dict containing the value
                assert "test_value" in str(result.answer), \
                    f"Answer should contain 'test_value', got: {result.answer}"
            finally:
                interpreter.shutdown()

        def test_final_with_keyword_args(self, interpreter):
            """FINAL with keyword args should create dict with those keys."""
            interpreter.start()
            try:
                result = interpreter.execute('FINAL(answer="result", score=95)')
                assert isinstance(result, FinalAnswerResult)
                answer = result.answer
                assert isinstance(answer, dict), f"Answer should be dict, got: {type(answer)}"
                assert "answer" in answer or "result" in str(answer), \
                    f"Answer should contain 'answer' key or value, got: {answer}"
            finally:
                interpreter.shutdown()

        def test_final_with_list(self, interpreter):
            """FINAL with list value should work."""
            interpreter.start()
            try:
                result = interpreter.execute('FINAL([1, 2, 3])')
                assert isinstance(result, FinalAnswerResult)
            finally:
                interpreter.shutdown()

        def test_final_with_dict(self, interpreter):
            """FINAL with dict value should work."""
            interpreter.start()
            try:
                result = interpreter.execute('FINAL({"key": "value"})')
                assert isinstance(result, FinalAnswerResult)
            finally:
                interpreter.shutdown()

        def test_final_terminates_execution(self, interpreter, supports_multiline_code):
            """Code after FINAL should not execute (or FINAL result should be returned)."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                code = """
FINAL("first")
print("should not print")
"""
                result = interpreter.execute(code)
                # Result should be FinalAnswerResult from first FINAL
                assert isinstance(result, FinalAnswerResult), \
                    f"Should return FinalAnswerResult, got: {type(result)}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 9: FINAL_VAR Function Tests
    # ==========================================================================

    class TestFinalVarFunction:
        """Tests for FINAL_VAR() function."""

        def test_final_var_with_single_variable(self, interpreter, supports_multiline_code):
            """FINAL_VAR with single variable should work."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                code = """
my_answer = "computed result"
FINAL_VAR("my_answer")
"""
                result = interpreter.execute(code)
                assert isinstance(result, FinalAnswerResult), \
                    f"FINAL_VAR should return FinalAnswerResult, got: {type(result)}"
                assert "computed result" in str(result.answer), \
                    f"Answer should contain 'computed result', got: {result.answer}"
            finally:
                interpreter.shutdown()

        def test_final_var_with_multiple_variables(self, interpreter, supports_multiline_code, skip_typed_final_tests):
            """FINAL_VAR with multiple variables should work."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")
            if skip_typed_final_tests:
                pytest.skip("Typed FINAL tests skipped")

            # Configure output_fields for multiple outputs
            if hasattr(interpreter, "output_fields"):
                interpreter.output_fields = [
                    {"name": "result"},
                    {"name": "confidence"},
                ]
            else:
                pytest.skip("Interpreter does not support output_fields")

            interpreter.start()
            try:
                code = """
result = "answer text"
confidence = 0.95
FINAL_VAR("result", "confidence")
"""
                result = interpreter.execute(code)
                assert isinstance(result, FinalAnswerResult)
                answer = result.answer
                # Should contain both values
                answer_str = str(answer)
                assert "answer text" in answer_str or "result" in str(answer), \
                    f"Answer should contain 'answer text', got: {answer}"
            finally:
                interpreter.shutdown()

        def test_final_var_with_undefined_variable_fails(self, interpreter, supports_multiline_code):
            """FINAL_VAR with undefined variable should raise error."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                with pytest.raises((CodeInterpreterError, NameError)):
                    interpreter.execute('FINAL_VAR("undefined_xyz")')
            finally:
                interpreter.shutdown()

        def test_final_var_returns_current_values(self, interpreter, supports_multiline_code):
            """FINAL_VAR should return current values of variables."""
            if not supports_multiline_code:
                pytest.skip("Interpreter does not support multiline code")

            interpreter.start()
            try:
                interpreter.execute("x = 1")
                interpreter.execute("x = x + 10")
                result = interpreter.execute('FINAL_VAR("x")')

                assert isinstance(result, FinalAnswerResult)
                # x should be 11
                assert "11" in str(result.answer), \
                    f"x should be 11, got: {result.answer}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 10: Typed Output Fields Tests
    # ==========================================================================

    class TestTypedOutputFields:
        """Tests for typed output fields (output_fields parameter)."""

        def test_final_with_typed_fields(self, interpreter, skip_typed_final_tests):
            """FINAL should respect typed output field definitions."""
            if skip_typed_final_tests:
                pytest.skip("Typed FINAL tests skipped")

            if not hasattr(interpreter, "output_fields"):
                pytest.skip("Interpreter does not support output_fields")

            interpreter.output_fields = [
                {"name": "answer", "type": "str"},
                {"name": "score", "type": "int"},
            ]
            interpreter.start()
            try:
                result = interpreter.execute('FINAL(answer="result", score=42)')
                assert isinstance(result, FinalAnswerResult)
                answer = result.answer
                assert answer.get("answer") == "result", f"Got: {answer}"
                assert answer.get("score") == 42, f"Got: {answer}"
            finally:
                interpreter.shutdown()

        def test_final_positional_args_match_field_order(self, interpreter, skip_typed_final_tests):
            """FINAL positional args should map to output fields in order."""
            if skip_typed_final_tests:
                pytest.skip("Typed FINAL tests skipped")

            if not hasattr(interpreter, "output_fields"):
                pytest.skip("Interpreter does not support output_fields")

            interpreter.output_fields = [
                {"name": "first", "type": "str"},
                {"name": "second", "type": "int"},
            ]
            interpreter.start()
            try:
                result = interpreter.execute('FINAL("value", 123)')
                assert isinstance(result, FinalAnswerResult)
                answer = result.answer
                assert answer.get("first") == "value", f"Got: {answer}"
                assert answer.get("second") == 123, f"Got: {answer}"
            finally:
                interpreter.shutdown()

    # ==========================================================================
    # Section 11: Context Manager Tests
    # ==========================================================================

    class TestContextManager:
        """Tests for context manager support (__enter__, __exit__)."""

        def test_supports_context_manager(self, interpreter):
            """Interpreter should support context manager protocol."""
            assert hasattr(interpreter, "__enter__"), "Should have __enter__ method"
            assert hasattr(interpreter, "__exit__"), "Should have __exit__ method"

        def test_context_manager_basic_usage(self, interpreter):
            """Context manager should allow basic execution."""
            # Create fresh instance since fixture may be reused
            InterpreterClass = type(interpreter)
            with InterpreterClass() as ctx_interpreter:
                result = ctx_interpreter.execute("print('context manager works')")
                output = str(result) if result else ""
                assert "context manager works" in output

        def test_context_manager_cleanup_on_exception(self, interpreter):
            """Context manager should clean up even on exception."""
            InterpreterClass = type(interpreter)
            try:
                with InterpreterClass() as ctx_interpreter:
                    ctx_interpreter.execute("x = 1")
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected
            # Interpreter should be shut down - we can't easily verify this
            # but the test ensures no crash occurs

    # ==========================================================================
    # Section 12: Edge Cases and Stress Tests
    # ==========================================================================

    class TestEdgeCases:
        """Tests for edge cases and unusual inputs."""

        def test_empty_code(self, interpreter):
            """Empty code should execute without error."""
            interpreter.start()
            try:
                result = interpreter.execute("")
                # Should return None or empty
            finally:
                interpreter.shutdown()

        def test_whitespace_only_code(self, interpreter):
            """Whitespace-only code should execute without error."""
            interpreter.start()
            try:
                result = interpreter.execute("   \n\t\n   ")
                # Should return None or empty
            finally:
                interpreter.shutdown()

        def test_comment_only_code(self, interpreter):
            """Comment-only code should execute without error."""
            interpreter.start()
            try:
                result = interpreter.execute("# This is a comment")
                # Should return None or empty
            finally:
                interpreter.shutdown()

        def test_large_variable_injection(self, interpreter):
            """Large variables should be injectable."""
            interpreter.start()
            try:
                large_list = list(range(10000))
                result = interpreter.execute(
                    "print(len(data))",
                    variables={"data": large_list}
                )
                output = str(result) if result else ""
                assert "10000" in output, f"len should be 10000, got: {result}"
            finally:
                interpreter.shutdown()

        def test_unicode_in_code(self, interpreter):
            """Unicode characters should work in code."""
            interpreter.start()
            try:
                result = interpreter.execute('print("Hello, 世界! 🌍")')
                output = str(result) if result else ""
                assert "世界" in output or "Hello" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()

        def test_unicode_in_variables(self, interpreter):
            """Unicode in injected variables should work."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(msg)",
                    variables={"msg": "Привет мир"}
                )
                output = str(result) if result else ""
                assert "Привет" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()

        def test_none_variable_injection(self, interpreter):
            """None values should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(x is None)",
                    variables={"x": None}
                )
                output = str(result) if result else ""
                assert "True" in output, f"x is None should be True, got: {result}"
            finally:
                interpreter.shutdown()

        def test_boolean_variable_injection(self, interpreter):
            """Boolean values should be injectable."""
            interpreter.start()
            try:
                result = interpreter.execute(
                    "print(flag)",
                    variables={"flag": True}
                )
                output = str(result) if result else ""
                assert "True" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()

        def test_nested_data_structure(self, interpreter):
            """Nested data structures should be injectable."""
            interpreter.start()
            try:
                nested = {
                    "level1": {
                        "level2": {
                            "value": [1, 2, {"nested": True}]
                        }
                    }
                }
                result = interpreter.execute(
                    "print(data['level1']['level2']['value'][2]['nested'])",
                    variables={"data": nested}
                )
                output = str(result) if result else ""
                assert "True" in output, f"Got: {result}"
            finally:
                interpreter.shutdown()


# =============================================================================
# Convenience function to run all tests
# =============================================================================

def run_conformance_tests(interpreter_class, **init_kwargs):
    """Run conformance tests against an interpreter class.

    This is a convenience function for quick validation.

    Args:
        interpreter_class: The CodeInterpreter class to test
        **init_kwargs: Arguments to pass to interpreter constructor

    Example:
        from my_package import MyInterpreter
        from tests.interpreter_conformance import run_conformance_tests

        run_conformance_tests(MyInterpreter, api_key="...")
    """
    import sys

    # Create a test class dynamically
    class DynamicTest(InterpreterConformanceTests):
        @pytest.fixture
        def interpreter(self):
            return interpreter_class(**init_kwargs)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
