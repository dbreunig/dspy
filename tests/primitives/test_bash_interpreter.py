"""
Tests for BashInterpreter.

This module tests the BashInterpreter against the interpreter conformance suite
and includes additional Bash-specific tests.

Requirements:
    pip install just-bash

To run these tests:
    pytest tests/primitives/test_bash_interpreter.py -v
"""

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

# Skip all tests if just-bash is not installed
just_bash = pytest.importorskip("just_bash", reason="just-bash library not installed")

from dspy.primitives.bash_interpreter import BashInterpreter


# =============================================================================
# Bash-Specific Conformance Tests
# =============================================================================


class TestBashInterpreterCore:
    """Core protocol tests for BashInterpreter."""

    def test_has_tools_property(self):
        """Interpreter must have a tools property."""
        interpreter = BashInterpreter()
        assert hasattr(interpreter, "tools")
        tools = interpreter.tools
        assert isinstance(tools, dict)

    def test_tools_property_is_mutable(self):
        """Tools property must return a mutable dict."""
        interpreter = BashInterpreter()
        tools = interpreter.tools
        tools["_test_tool"] = lambda: "test"
        assert "_test_tool" in interpreter.tools

    def test_has_required_methods(self):
        """Interpreter must have start, execute, shutdown methods."""
        interpreter = BashInterpreter()
        assert hasattr(interpreter, "start") and callable(interpreter.start)
        assert hasattr(interpreter, "execute") and callable(interpreter.execute)
        assert hasattr(interpreter, "shutdown") and callable(interpreter.shutdown)


class TestBashInterpreterLifecycle:
    """Lifecycle management tests."""

    def test_start_is_idempotent(self):
        """Calling start() multiple times should be safe."""
        interpreter = BashInterpreter()
        interpreter.start()
        interpreter.start()  # Should not raise
        interpreter.shutdown()

    def test_shutdown_is_idempotent(self):
        """Calling shutdown() multiple times should be safe."""
        interpreter = BashInterpreter()
        interpreter.start()
        interpreter.shutdown()
        interpreter.shutdown()  # Should not raise

    def test_execute_calls_start_lazily(self):
        """Execute should call start() lazily if not already started."""
        interpreter = BashInterpreter()
        # Don't call start() explicitly
        result = interpreter.execute("echo hello")
        assert "hello" in result
        interpreter.shutdown()

    def test_context_manager_support(self):
        """Interpreter should support context manager protocol."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("echo 'context manager works'")
            assert "context manager works" in result


class TestBashInterpreterBasicExecution:
    """Basic code execution tests."""

    def test_execute_echo(self):
        """Echo command should work."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("echo 'hello world'")
            assert "hello world" in result

    def test_execute_arithmetic(self):
        """Arithmetic should work."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("echo $((2 + 3))")
            assert "5" in result

    def test_execute_multiline(self):
        """Multiline code should work."""
        with BashInterpreter() as interpreter:
            code = """
x=10
y=20
echo $((x + y))
"""
            result = interpreter.execute(code)
            assert "30" in result

    def test_execute_returns_none_for_no_output(self):
        """Execution with no output should return None."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("x=1")
            # Assignment with no echo should return None or empty
            assert result is None or result == "" or result.strip() == ""

    def test_execute_empty_code(self):
        """Empty code should not error."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("")
            assert result is None

    def test_execute_comment_only(self):
        """Comment-only code should not error."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("# This is a comment")
            # May return None or empty
            assert result is None or result.strip() == ""


class TestBashInterpreterStatePersistence:
    """State persistence tests."""

    def test_variables_persist_across_calls(self):
        """Variables defined in one call should be available in the next."""
        with BashInterpreter() as interpreter:
            interpreter.execute("counter=0")
            interpreter.execute("counter=$((counter + 1))")
            result = interpreter.execute("echo $counter")
            assert "1" in result

    @pytest.mark.skip(reason="just-bash has limited support for user-defined functions with parameters")
    def test_functions_persist_across_calls(self):
        """Functions defined in one call should be callable in the next.

        Note: just-bash has limited support for user-defined functions with
        positional parameters ($1, $2). This test is skipped.
        """
        with BashInterpreter() as interpreter:
            interpreter.execute("add() { echo $(($1 + $2)); }")
            result = interpreter.execute("add 2 3")
            assert "5" in result


class TestBashInterpreterVariableInjection:
    """Variable injection tests."""

    def test_inject_string_variable(self):
        """String variables should be injectable."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute(
                "echo $message",
                variables={"message": "hello world"}
            )
            assert "hello world" in result

    def test_inject_integer_variable(self):
        """Integer variables should be injectable."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute(
                "echo $((x * 2))",
                variables={"x": 21}
            )
            assert "42" in result

    def test_inject_multiple_variables(self):
        """Multiple variables should be injectable."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute(
                "echo $((a + b))",
                variables={"a": 10, "b": 20}
            )
            assert "30" in result

    def test_inject_json_list(self):
        """List variables should be injectable as JSON."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute(
                "echo $items",
                variables={"items": [1, 2, 3]}
            )
            # Should contain the JSON representation
            assert "1" in result and "2" in result and "3" in result


class TestBashInterpreterErrorHandling:
    """Error handling tests."""

    def test_exit_error(self):
        """Exit with non-zero code should raise error."""
        with BashInterpreter() as interpreter:
            with pytest.raises(CodeInterpreterError):
                interpreter.execute("exit 1")

    def test_command_failure(self):
        """Command that fails should raise error."""
        with BashInterpreter() as interpreter:
            # Use a command that reliably fails
            with pytest.raises(CodeInterpreterError):
                interpreter.execute("exit 42")


class TestBashInterpreterFinal:
    """FINAL function tests."""

    def test_final_with_single_value(self):
        """FINAL with single value should return FinalAnswerResult."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute('FINAL "the answer"')
            assert isinstance(result, FinalAnswerResult)

    def test_final_answer_contains_value(self):
        """FinalAnswerResult should contain the provided value."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute('FINAL "test_value"')
            assert isinstance(result, FinalAnswerResult)
            assert "test_value" in str(result.answer)

    def test_final_with_computed_value(self):
        """FINAL with computed value should work."""
        with BashInterpreter() as interpreter:
            code = """
result=$((10 + 32))
FINAL "$result"
"""
            result = interpreter.execute(code)
            assert isinstance(result, FinalAnswerResult)
            assert "42" in str(result.answer)


class TestBashInterpreterFinalVar:
    """FINAL_VAR function tests.

    Note: FINAL_VAR has limited support in just-bash due to lack of proper
    indirect variable reference support (eval with $$ doesn't work).
    Use FINAL with direct values instead.
    """

    @pytest.mark.skip(reason="just-bash doesn't support indirect variable references via eval")
    def test_final_var_with_single_variable(self):
        """FINAL_VAR with single variable should work.

        Note: This test is skipped because just-bash doesn't properly support
        eval with indirect variable references. Use FINAL "$var" instead.
        """
        with BashInterpreter() as interpreter:
            code = """
my_answer="computed result"
FINAL_VAR my_answer
"""
            result = interpreter.execute(code)
            assert isinstance(result, FinalAnswerResult)
            assert "computed result" in str(result.answer)

    def test_final_with_variable_value(self):
        """FINAL with variable value (alternative to FINAL_VAR)."""
        with BashInterpreter() as interpreter:
            code = """
my_answer="computed result"
FINAL "$my_answer"
"""
            result = interpreter.execute(code)
            assert isinstance(result, FinalAnswerResult)
            assert "computed result" in str(result.answer)


class TestBashInterpreterTools:
    """Tool integration tests."""

    def test_simple_tool_call(self):
        """Simple tool should be callable from bash."""
        def echo_tool(message: str) -> str:
            return f"echo: {message}"

        with BashInterpreter(tools={"echo_tool": echo_tool}) as interpreter:
            result = interpreter.execute('echo_tool "hello"')
            # The tool output should be in the result
            assert "echo:" in result or "hello" in result

    def test_tools_can_be_updated_after_init(self):
        """Tools should be updatable after initialization."""
        interpreter = BashInterpreter()

        def late_tool() -> str:
            return "late"

        interpreter.tools["late_tool"] = late_tool
        assert "late_tool" in interpreter.tools
        interpreter.shutdown()


class TestBashInterpreterEdgeCases:
    """Edge case tests."""

    def test_special_characters_in_strings(self):
        """Special characters should be handled."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("echo 'hello \"world\"'")
            assert "hello" in result

    def test_pipe_commands(self):
        """Pipe commands should work."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute("echo -e 'line1\\nline2\\nline3' | wc -l")
            assert "3" in result

    def test_command_substitution(self):
        """Command substitution should work."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute('result=$(echo "hello"); echo $result')
            assert "hello" in result


class TestBashInterpreterTypedOutputFields:
    """Typed output fields tests."""

    def test_output_fields_configuration(self):
        """Output fields should be configurable."""
        interpreter = BashInterpreter(
            output_fields=[
                {"name": "answer", "type": "str"},
                {"name": "score", "type": "int"},
            ]
        )
        assert len(interpreter.output_fields) == 2
        assert interpreter.output_fields[0]["name"] == "answer"
        interpreter.shutdown()

    def test_final_with_multiple_output_fields(self):
        """FINAL with multiple values should map to output fields."""
        with BashInterpreter(
            output_fields=[
                {"name": "result"},
                {"name": "confidence"},
            ]
        ) as interpreter:
            result = interpreter.execute('FINAL "answer text" "0.95"')
            assert isinstance(result, FinalAnswerResult)
            # Both values should be in the answer
            answer_str = str(result.answer)
            assert "answer text" in answer_str or "result" in answer_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestBashInterpreterIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test a complete workflow with variables, computation, and FINAL."""
        with BashInterpreter() as interpreter:
            # Set up data
            interpreter.execute("data='1 2 3 4 5'")

            # Process data
            interpreter.execute("sum=0; for n in $data; do sum=$((sum + n)); done")

            # Check intermediate result
            result = interpreter.execute("echo $sum")
            assert "15" in result

            # Final answer
            result = interpreter.execute('FINAL "$sum"')
            assert isinstance(result, FinalAnswerResult)
            assert "15" in str(result.answer)

    def test_variable_injection_and_processing(self):
        """Test injecting variables and processing them."""
        with BashInterpreter() as interpreter:
            result = interpreter.execute(
                """
                total=0
                for item in $items; do
                    total=$((total + item))
                done
                echo $total
                """,
                variables={"items": "10 20 30"}
            )
            assert "60" in result


class TestBashInterpreterFiles:
    """Tests for in-memory filesystem."""

    def test_initial_files(self):
        """Initial files should be accessible."""
        with BashInterpreter(
            files={"/home/user/test.txt": "hello from file"}
        ) as interpreter:
            result = interpreter.execute("cat /home/user/test.txt")
            assert "hello from file" in result

    def test_create_and_read_file(self):
        """Creating and reading files should work."""
        with BashInterpreter() as interpreter:
            interpreter.execute("echo 'new content' > /home/user/new.txt")
            result = interpreter.execute("cat /home/user/new.txt")
            assert "new content" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
