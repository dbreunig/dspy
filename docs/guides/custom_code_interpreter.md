# Building Custom Code Interpreters for DSPy RLM

This guide explains how to create custom `CodeInterpreter` implementations for use with DSPy's Recursive Language Model (RLM) module. Custom interpreters allow you to use lightweight environments (emulated sandboxes), specialized interpreters (remote sandboxes like E2B, Modal, Vercel), or any other code execution environment.

## Table of Contents

- [Overview](#overview)
- [Protocol Interface](#protocol-interface)
- [Required Behaviors](#required-behaviors)
- [Error Handling](#error-handling)
- [Tools Integration](#tools-integration)
- [FINAL and FINAL_VAR Functions](#final-and-final_var-functions)
- [Implementation Checklist](#implementation-checklist)
- [Example Implementations](#example-implementations)
- [Testing Your Implementation](#testing-your-implementation)
- [Integration with RLM](#integration-with-rlm)
- [Thread Safety Considerations](#thread-safety-considerations)

## Overview

DSPy's RLM module uses a `CodeInterpreter` to execute Python code in a sandboxed environment. The interpreter allows the LLM to:

1. Execute Python code iteratively
2. Maintain state across multiple execution calls
3. Call host-side tools (functions) from within the sandbox
4. Signal completion via `FINAL()` or `FINAL_VAR()` functions

The `CodeInterpreter` protocol uses duck typing, meaning any class that implements the required methods will work with RLM. DSPy provides a `@runtime_checkable` Protocol class for optional type checking.

## Protocol Interface

Your interpreter must implement the following interface:

```python
from typing import Any, Callable

class MyInterpreter:
    """Custom CodeInterpreter implementation."""

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        """Tools available for interpreter code to call.

        Returns:
            Dictionary mapping tool names to callable functions.
            Each tool accepts keyword arguments and returns a string.
        """
        ...

    def start(self) -> None:
        """Initialize the interpreter and allocate resources.

        This method prepares the interpreter for code execution. It can be called
        explicitly to pre-warm the interpreter, or implementations may call it
        lazily on first execute().

        Must be idempotent - safe to call multiple times.
        """
        ...

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code and return the result.

        Args:
            code: Python code to execute
            variables: Variables to inject into the namespace before execution.
                      These are available as top-level variables in the code.

        Returns:
            One of:
            - FinalAnswerResult: If FINAL() or FINAL_VAR() was called in code
            - str: Captured stdout from print() statements
            - list: Multiple output lines
            - None: If no output was produced

        Raises:
            CodeInterpreterError: On runtime errors (undefined vars, tool failures, etc.)
            SyntaxError: On invalid Python syntax
        """
        ...

    def shutdown(self) -> None:
        """Release resources and terminate the interpreter session.

        After shutdown(), the interpreter should not be used again.
        A new instance should be created for a fresh session.
        """
        ...
```

### Importing Protocol Types

```python
from dspy.primitives.code_interpreter import (
    CodeInterpreter,        # Protocol class (for type hints)
    CodeInterpreterError,   # Exception for runtime errors
    FinalAnswerResult,      # Return type for FINAL()/FINAL_VAR()
)
```

## Required Behaviors

### 1. State Persistence

Variables and state **must persist** across multiple `execute()` calls within a session. This allows the LLM to build up results incrementally:

```python
# First call
interpreter.execute("x = 10")

# Second call - x should still be available
result = interpreter.execute("print(x + 5)")  # Should print "15"
```

### 2. Variable Injection

When `variables` are provided, they must be available as top-level variables in the code:

```python
result = interpreter.execute(
    "print(context[:100])",
    variables={"context": "A very long string..."}
)
```

Variables should be serializable to JSON for cross-environment transport. DSPy handles serialization of common Python types (tuple → list, set → list).

### 3. Output Capture

Your interpreter should capture and return output from:
- `print()` statements → return as `str`
- Expression evaluation (last expression value) → return the value
- Multiple outputs → return as `list` or concatenated `str`

### 4. Lazy Initialization

If `start()` has not been called before `execute()`, your implementation should call it lazily:

```python
def execute(self, code, variables=None):
    if not self._started:
        self.start()
    # ... execute code
```

### 5. Idempotent Methods

Both `start()` and `shutdown()` should be idempotent (safe to call multiple times):

```python
interpreter.start()
interpreter.start()  # Should not error or double-allocate resources

interpreter.shutdown()
interpreter.shutdown()  # Should not error
```

### 6. Context Manager Support (Recommended)

Supporting the context manager protocol allows clean resource management:

```python
class MyInterpreter:
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()

# Usage
with MyInterpreter() as interpreter:
    result = interpreter.execute("print('hello')")
```

## Error Handling

### CodeInterpreterError

Raise `CodeInterpreterError` for runtime errors:

```python
from dspy.primitives.code_interpreter import CodeInterpreterError

class MyInterpreter:
    def execute(self, code, variables=None):
        try:
            # Execute code
            ...
        except NameError as e:
            raise CodeInterpreterError(f"NameError: {e}")
        except Exception as e:
            raise CodeInterpreterError(f"{type(e).__name__}: {e}")
```

Use `CodeInterpreterError` for:
- Runtime errors (NameError, TypeError, ValueError, etc.)
- Undefined variable access
- Tool call failures
- Resource limits exceeded (memory, time)
- Any other execution-time errors

### SyntaxError

Raise the standard Python `SyntaxError` for invalid syntax:

```python
def execute(self, code, variables=None):
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax: {e}")
    # ... execute compiled code
```

### Error Message Format

RLM displays errors to the LLM, so error messages should be clear and actionable:

```python
# Good error messages
raise CodeInterpreterError("NameError: name 'undefined_var' is not defined")
raise CodeInterpreterError("Tool 'search' failed: connection timeout")

# Less helpful
raise CodeInterpreterError("execution failed")
```

## Tools Integration

Tools are host-side functions that can be called from within the sandbox. RLM automatically provides `llm_query` and `llm_query_batched` tools, plus any user-defined tools.

### Tools Property

Your interpreter must expose a `tools` property that returns a mutable dict:

```python
class MyInterpreter:
    def __init__(self, tools=None):
        self._tools = dict(tools or {})

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools
```

**Important**: RLM updates the tools dict with fresh `llm_query` tools on each `forward()` call:

```python
# RLM does this internally
interpreter.tools.update({"llm_query": ..., "llm_query_batched": ...})
```

### Tool Execution

When code calls a tool, your interpreter should:

1. Intercept the call (via AST transformation, proxy objects, or message passing)
2. Execute the host-side function
3. Return the result to the sandbox

```python
# Inside sandbox, user writes:
result = my_tool(query="search term", limit=10)

# Your interpreter should:
# 1. Detect the call to my_tool
# 2. Call self.tools["my_tool"](query="search term", limit=10)
# 3. Return the result string to the sandbox
```

### Tool Requirements

- Tools are called with keyword arguments (and optionally positional)
- Tools return strings
- Tool names must be valid Python identifiers
- Reserved names: `llm_query`, `llm_query_batched`, `FINAL`, `FINAL_VAR`, `print`

## FINAL and FINAL_VAR Functions

These functions signal that the LLM has produced its final answer.

### FINAL Function

`FINAL()` accepts the output values and returns a `FinalAnswerResult`:

```python
# Single output field named "answer" (default)
FINAL("the answer")
# → FinalAnswerResult(answer={"answer": "the answer"})

# Multiple output fields
FINAL(answer="the answer", confidence=0.95)
# → FinalAnswerResult(answer={"answer": "the answer", "confidence": 0.95})

# Positional args (mapped to output field names in order)
FINAL("the answer", 0.95)
# → FinalAnswerResult(answer={"answer": "the answer", "confidence": 0.95})
```

### FINAL_VAR Function

`FINAL_VAR()` takes variable names as strings and returns their values:

```python
# In sandbox code:
result = "computed answer"
score = 42
FINAL_VAR("result", "score")
# → FinalAnswerResult(answer={"answer": "computed answer", "score": 42})
```

### Implementing FINAL/FINAL_VAR

Your interpreter must:

1. Register `FINAL` and `FINAL_VAR` as callable functions in the sandbox
2. When called, return a `FinalAnswerResult` from `execute()`
3. Support the `output_fields` parameter to define typed outputs

```python
from dspy.primitives.code_interpreter import FinalAnswerResult

class MyInterpreter:
    def __init__(self, output_fields=None, **kwargs):
        # output_fields: [{"name": "answer", "type": "str"}, ...]
        self.output_fields = output_fields or [{"name": "answer"}]

    def _make_final_function(self):
        output_fields = self.output_fields

        def FINAL(*args, **kwargs):
            # Map positional args to field names
            if args:
                for i, (field, value) in enumerate(zip(output_fields, args)):
                    kwargs[field["name"]] = value
            return FinalAnswerResult(kwargs)

        return FINAL
```

## Implementation Checklist

Use this checklist to ensure your interpreter meets all requirements:

### Core Protocol
- [ ] `tools` property returns `dict[str, Callable[..., str]]`
- [ ] `tools` property returns a mutable dict (RLM updates it)
- [ ] `start()` method initializes resources
- [ ] `start()` is idempotent
- [ ] `execute(code, variables)` runs Python code
- [ ] `execute()` calls `start()` lazily if needed
- [ ] `shutdown()` method releases resources
- [ ] `shutdown()` is idempotent

### State Management
- [ ] Variables persist across `execute()` calls
- [ ] Injected variables are available in code
- [ ] State is isolated per interpreter instance

### Output Handling
- [ ] `print()` output is captured and returned as `str`
- [ ] Expression values can be returned
- [ ] `FinalAnswerResult` is returned when `FINAL()`/`FINAL_VAR()` called
- [ ] `None` is returned when no output produced

### Error Handling
- [ ] `CodeInterpreterError` raised for runtime errors
- [ ] `SyntaxError` raised for invalid Python syntax
- [ ] Error messages are clear and include error type

### Tools
- [ ] Tools can be called from sandbox code
- [ ] Tools support keyword arguments
- [ ] Tools support positional arguments
- [ ] Tool return values are accessible in sandbox

### FINAL Functions
- [ ] `FINAL()` with keyword args works
- [ ] `FINAL()` with positional args works
- [ ] `FINAL_VAR()` with variable names works
- [ ] Output field types are respected (when `output_fields` provided)

### Resource Management
- [ ] Context manager support (`__enter__`, `__exit__`)
- [ ] Resources are properly cleaned up on `shutdown()`

## Example Implementations

### Minimal Example

Here's a minimal (but functional) interpreter using Python's `exec()`:

```python
from typing import Any, Callable
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

class SimpleInterpreter:
    """Minimal interpreter using Python exec() - NOT SANDBOXED."""

    def __init__(self, tools=None, output_fields=None):
        self._tools = dict(tools or {})
        self.output_fields = output_fields or [{"name": "answer"}]
        self._namespace = {}
        self._started = False

    @property
    def tools(self):
        return self._tools

    def start(self):
        if self._started:
            return
        self._namespace = {"__builtins__": __builtins__}
        self._register_builtins()
        self._started = True

    def _register_builtins(self):
        # Capture output
        self._output = []
        self._final_result = None

        def capture_print(*args, **kwargs):
            self._output.append(" ".join(str(a) for a in args))

        def FINAL(*args, **kwargs):
            if args:
                fields = self.output_fields
                for i, value in enumerate(args):
                    if i < len(fields):
                        kwargs[fields[i]["name"]] = value
            self._final_result = FinalAnswerResult(kwargs)

        def FINAL_VAR(*var_names):
            result = {}
            fields = self.output_fields
            for i, name in enumerate(var_names):
                if name not in self._namespace:
                    raise NameError(f"Variable '{name}' is not defined")
                if i < len(fields):
                    result[fields[i]["name"]] = self._namespace[name]
            self._final_result = FinalAnswerResult(result)

        self._namespace["print"] = capture_print
        self._namespace["FINAL"] = FINAL
        self._namespace["FINAL_VAR"] = FINAL_VAR

        # Register tools
        for name, func in self._tools.items():
            self._namespace[name] = func

    def execute(self, code, variables=None):
        if not self._started:
            self.start()

        # Reset output state
        self._output = []
        self._final_result = None

        # Inject variables
        if variables:
            self._namespace.update(variables)

        # Re-register tools (in case they were updated)
        for name, func in self._tools.items():
            self._namespace[name] = func

        # Check syntax
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax: {e}")

        # Execute
        try:
            exec(code, self._namespace)
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
```

### Remote Sandbox Example (E2B-style)

Here's a skeleton for a remote sandbox implementation:

```python
from typing import Any, Callable
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

class RemoteSandboxInterpreter:
    """Interpreter using a remote sandbox service."""

    def __init__(self, api_key=None, tools=None, output_fields=None):
        self._api_key = api_key
        self._tools = dict(tools or {})
        self.output_fields = output_fields or [{"name": "answer"}]
        self._session_id = None

    @property
    def tools(self):
        return self._tools

    def start(self):
        if self._session_id:
            return
        # Create remote sandbox session
        self._session_id = self._create_session()
        # Register FINAL/FINAL_VAR in remote sandbox
        self._setup_sandbox()

    def _create_session(self):
        # Call remote API to create sandbox
        # return session_id
        ...

    def _setup_sandbox(self):
        # Register tools and FINAL functions in remote sandbox
        ...

    def execute(self, code, variables=None):
        if not self._session_id:
            self.start()

        # Send code to remote sandbox
        response = self._send_code(code, variables)

        # Parse response
        if response.get("error"):
            error_type = response.get("error_type", "Error")
            if error_type == "SyntaxError":
                raise SyntaxError(response["error"])
            raise CodeInterpreterError(response["error"])

        if response.get("final_answer"):
            return FinalAnswerResult(response["final_answer"])

        return response.get("output")

    def _send_code(self, code, variables):
        # Call remote API to execute code
        # Handle tool callbacks if needed
        ...

    def shutdown(self):
        if self._session_id:
            # Terminate remote session
            self._terminate_session()
            self._session_id = None

    def _terminate_session(self):
        # Call remote API to terminate sandbox
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
```

## Testing Your Implementation

DSPy provides a comprehensive test suite to validate your interpreter. Run the test suite against your implementation:

```python
# tests/test_custom_interpreter.py
import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

# Import your interpreter
from my_package import MyInterpreter

# Use the DSPy test suite
from tests.interpreter_conformance import InterpreterConformanceTests

class TestMyInterpreter(InterpreterConformanceTests):
    """Run conformance tests against MyInterpreter."""

    @pytest.fixture
    def interpreter(self):
        """Provide your interpreter instance."""
        return MyInterpreter()
```

See `tests/interpreter_conformance.py` for the full test suite.

### Key Test Categories

1. **Basic Execution** - Simple code runs correctly
2. **State Persistence** - Variables persist across calls
3. **Variable Injection** - Injected variables are accessible
4. **Error Handling** - Proper exceptions raised
5. **Tool Execution** - Tools can be called with args/kwargs
6. **FINAL Functions** - FINAL and FINAL_VAR work correctly
7. **Lifecycle** - start/shutdown behave correctly

## Integration with RLM

Once your interpreter is implemented and tested, use it with RLM:

```python
import dspy
from my_package import MyInterpreter

# Create interpreter instance
interpreter = MyInterpreter(
    api_key="...",  # if needed
)

# Use with RLM
rlm = dspy.RLM(
    "context, query -> answer",
    interpreter=interpreter,
    max_iterations=10,
)

# Execute
result = rlm(
    context="Very long document...",
    query="What is the main topic?"
)
print(result.answer)
```

### Important Notes

1. **RLM Updates Tools**: RLM calls `interpreter.tools.update(...)` with `llm_query` and `llm_query_batched` on each `forward()` call. Your `tools` property must return a mutable dict.

2. **Output Fields**: RLM may set `interpreter.output_fields` if your interpreter has that attribute. This enables typed FINAL signatures.

3. **Single Instance Sharing**: If you provide a single interpreter instance, RLM reuses it across `forward()` calls. This is efficient but means:
   - State accumulates across calls (may need manual cleanup)
   - Not thread-safe for concurrent use

4. **No Interpreter Provided**: If you don't provide an interpreter, RLM creates a fresh `PythonInterpreter` for each `forward()` call and shuts it down afterward.

## Thread Safety Considerations

RLM instances are **not thread-safe** when using a custom interpreter. For concurrent use:

```python
# Option 1: Create separate RLM instances
def worker(query):
    rlm = dspy.RLM("query -> answer", interpreter=MyInterpreter())
    return rlm(query=query)

# Option 2: Use the default (no custom interpreter)
# RLM creates fresh PythonInterpreter per forward() call
rlm = dspy.RLM("query -> answer")  # Thread-safe

# Option 3: Interpreter pooling
from queue import Queue

interpreter_pool = Queue()
for _ in range(4):
    interp = MyInterpreter()
    interp.start()  # Pre-warm
    interpreter_pool.put(interp)

def worker(query):
    interp = interpreter_pool.get()
    try:
        rlm = dspy.RLM("query -> answer", interpreter=interp)
        return rlm(query=query)
    finally:
        interpreter_pool.put(interp)
```

## Non-Python Interpreters

While DSPy's `PythonInterpreter` executes Python code, you can implement interpreters for other languages. The protocol is language-agnostic - your interpreter just needs to:

1. Execute code in some language
2. Support variable injection
3. Provide tools callable from the code
4. Support FINAL/FINAL_VAR for signaling completion

### Input Variables as Files

For interpreters where processing large text inputs is common (e.g., bash, shell scripts), consider storing input variables as **files** rather than as code-level variables. This approach:

- Enables use of standard text processing tools (grep, awk, sed, etc.)
- Avoids escaping issues with large text in code
- Separates data from code

Example approach:
```
# Store input as file: /input/context contains the text
# Set shell variable to path: $context = "/input/context"
# LLM can process with: grep "pattern" $context | wc -l
```

### BashInterpreter Example

The [dspy-bash-interpreter](https://github.com/dspy-community/dspy-bash-interpreter) library provides a bash interpreter that implements this protocol:

```python
from dspy_bash_interpreter import BashInterpreter

interpreter = BashInterpreter()
rlm = dspy.RLM(
    "log_content: str -> error_count: int",
    interpreter=interpreter,
)

# log_content is stored as /input/log_content file
# LLM can use: grep "ERROR" $log_content | wc -l
result = rlm(log_content=log_text)
```

Key features:
- Bash code execution via just-bash (pure Python, sandboxed)
- Input variables stored as files in `/input/` directory
- Shell variables point to file paths (e.g., `$context` = `/input/context`)
- Full support for Unix tools: grep, awk, sed, head, tail, wc, etc.

## Additional Resources

- [DSPy Documentation](https://dspy.ai)
- [RLM Paper](https://arxiv.org/abs/...) - "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
- [CodeInterpreter Protocol](../dspy/primitives/code_interpreter.py)
- [PythonInterpreter Reference Implementation](../dspy/primitives/python_interpreter.py)
- [Interpreter Conformance Tests](../tests/interpreter_conformance.py)
- [BashInterpreter](https://github.com/dspy-community/dspy-bash-interpreter) - Bash interpreter example

## Support

If you have questions or need help implementing a custom interpreter:

1. Check existing implementations in `dspy/primitives/`
2. Run the conformance test suite to identify issues
3. Open a discussion on GitHub
