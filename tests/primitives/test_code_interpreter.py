import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError


def test_code_interpreter_error_is_dspy_error():
    error = CodeInterpreterError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, RuntimeError)


def test_code_execution_error_is_dspy_error():
    error = CodeExecutionError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, CodeInterpreterError)
