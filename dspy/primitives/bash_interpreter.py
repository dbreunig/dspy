"""
Bash code interpreter for DSPy RLM.

This module provides a BashInterpreter that executes Bash scripts using the
just-bash library (a pure Python bash interpreter with in-memory filesystem),
along with a BashRLM subclass that adapts RLM for Bash code generation.

Installation:
    pip install just-bash

Example:
    ```python
    from dspy.primitives.bash_interpreter import BashInterpreter, BashRLM

    # Use BashInterpreter directly
    with BashInterpreter() as interp:
        result = interp.execute('echo "Hello, World!"')
        print(result)  # Hello, World!

    # Use BashRLM for LLM-driven bash code generation
    rlm = BashRLM("files: list[str] -> total_size: int")
    result = rlm(files=["/etc/passwd", "/etc/hosts"])
    ```
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalAnswerResult

# Marker used to detect FINAL calls in bash output
_FINAL_MARKER = "__DSPY_FINAL__"
_FINAL_VAR_MARKER = "__DSPY_FINAL_VAR__"

__all__ = ["BashInterpreter", "BashRLM"]


class BashInterpreter:
    """Bash code interpreter implementing the CodeInterpreter protocol.

    Executes Bash scripts using the just-bash library, which provides a pure
    Python bash interpreter with an in-memory virtual filesystem. State
    persists across multiple execute() calls.

    Features:
        - Variable injection via shell variables
        - Tool support via bash functions
        - FINAL() for signaling completion with values
        - State persistence across calls (variables)
        - In-memory sandboxed filesystem

    Known Limitations (due to just-bash):
        - User-defined functions with positional parameters ($1, $2) have
          limited support. Use variables instead of function parameters.
        - FINAL_VAR() has limited support due to lack of indirect variable
          references. Use FINAL "$var" instead of FINAL_VAR var.
        - Some advanced bash features may not be supported.

    Example:
        ```python
        with BashInterpreter() as interp:
            interp.execute('count=0')
            interp.execute('count=$((count + 1))')
            result = interp.execute('echo $count')
            print(result)  # "1"

            # Use FINAL with variable value
            result = interp.execute('FINAL "$count"')
        ```
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        env: dict[str, str] | None = None,
        files: dict[str, str | bytes] | None = None,
        cwd: str = "/home/user",
    ):
        """Initialize the Bash interpreter.

        Args:
            tools: Dictionary of tool functions callable from bash scripts.
                   Each tool is exposed as a bash function.
            output_fields: List of output field definitions for FINAL/FINAL_VAR.
                          Format: [{"name": "field_name", "type": "str"}, ...]
            env: Additional environment variables to set.
            files: Initial files to create in the virtual filesystem.
                   Format: {"/path/to/file": "contents", ...}
            cwd: Initial working directory. Defaults to "/home/user".
        """
        self._tools = dict(tools or {})
        self.output_fields = output_fields or [{"name": "answer"}]
        self._env = dict(env or {})
        self._files = dict(files or {})
        self._cwd = cwd
        self._started = False
        self._shutdown = False
        self._bash = None
        self._loop = None

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        """Tools available for bash code to call."""
        return self._tools

    def start(self) -> None:
        """Initialize the interpreter session.

        Creates the just-bash Bash instance with configured environment.
        This method is idempotent - safe to call multiple times.
        """
        if self._started and not self._shutdown:
            return

        try:
            from just_bash import Bash
        except ImportError as e:
            raise CodeInterpreterError(
                "just-bash library not installed. Install with: pip install just-bash"
            ) from e

        # Create the Bash interpreter instance
        self._bash = Bash(
            env=self._env,
            files=self._files,
            cwd=self._cwd,
            errexit=False,  # Don't exit on error - we handle errors ourselves
            pipefail=False,
            nounset=False,
        )

        self._started = True
        self._shutdown = False

    def _get_event_loop(self):
        """Get or create an event loop for running async code."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use nest_asyncio or run in thread
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running loop
            loop = self._get_event_loop()
            return loop.run_until_complete(coro)

    def _build_final_functions(self) -> str:
        """Generate FINAL and FINAL_VAR bash functions.

        Uses a simpler approach compatible with just-bash's bash implementation.
        """
        field_names = [f["name"] for f in self.output_fields]

        # Build simpler FINAL function - outputs field:value pairs
        # Format: __DSPY_FINAL__field1=value1|field2=value2
        final_func = f'''
FINAL() {{
    local result=""
    local i=1
    for arg in "$@"; do
        case $i in'''

        for idx, field_name in enumerate(field_names):
            final_func += f'''
            {idx + 1}) result="${{result}}{field_name}=$arg|" ;;'''

        final_func += f'''
            *) result="${{result}}arg_$i=$arg|" ;;
        esac
        i=$((i + 1))
    done
    echo "{_FINAL_MARKER}$result"
}}
'''

        # Build FINAL_VAR function
        final_var_func = f'''
FINAL_VAR() {{
    local result=""
    local i=1
    for var_name in "$@"; do
        eval "local var_value=\\$$var_name"
        case $i in'''

        for idx, field_name in enumerate(field_names):
            final_var_func += f'''
            {idx + 1}) result="${{result}}{field_name}=$var_value|" ;;'''

        final_var_func += f'''
            *) result="${{result}}$var_name=$var_value|" ;;
        esac
        i=$((i + 1))
    done
    echo "{_FINAL_VAR_MARKER}$result"
}}
'''
        return final_func + "\n" + final_var_func

    def _build_tool_functions(self) -> str:
        """Generate bash functions for tools.

        Tools are implemented by outputting a special marker that we intercept
        in the output parsing phase.
        """
        functions = []
        for name in self._tools:
            # Create a function that outputs a marker for tool calls
            func = f'''
{name}() {{
    echo "{_FINAL_MARKER}__TOOL__:{name}:$*"
}}
'''
            functions.append(func)
        return "\n".join(functions)

    def _serialize_variable(self, name: str, value: Any) -> str:
        """Convert a Python variable to a bash variable assignment."""
        if value is None:
            return f'{name}=""'
        elif isinstance(value, bool):
            return f'{name}={"true" if value else "false"}'
        elif isinstance(value, (int, float)):
            return f"{name}={value}"
        elif isinstance(value, str):
            # Escape single quotes for bash
            escaped = value.replace("'", "'\"'\"'")
            return f"{name}='{escaped}'"
        elif isinstance(value, (list, dict)):
            # Store complex types as JSON strings
            json_str = json.dumps(value).replace("'", "'\"'\"'")
            return f"{name}='{json_str}'"
        else:
            # Convert to string
            str_val = str(value).replace("'", "'\"'\"'")
            return f"{name}='{str_val}'"

    def _parse_final_output(self, data_str: str) -> dict:
        """Parse the simple field=value|field=value| format from FINAL output."""
        result = {}
        # Remove trailing pipe if present
        data_str = data_str.rstrip("|")
        if not data_str:
            return result

        # Split by pipe and parse each field=value pair
        for pair in data_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key.strip()] = value
        return result

    def _parse_output(self, output: str) -> tuple[str | FinalAnswerResult, list[tuple[str, str]]]:
        """Parse bash output for FINAL markers and tool calls.

        Returns:
            Tuple of (result, tool_calls) where result is either the cleaned
            output string or a FinalAnswerResult.
        """
        lines = output.split("\n")
        clean_lines = []
        tool_calls = []

        for line in lines:
            # Check for tool call marker
            if f"{_FINAL_MARKER}__TOOL__:" in line:
                match = re.search(rf"{_FINAL_MARKER}__TOOL__:(\w+):(.*)", line)
                if match:
                    tool_calls.append((match.group(1), match.group(2)))
                continue

            # Check for FINAL_VAR marker (simple format: field=value|field=value|)
            if _FINAL_VAR_MARKER in line:
                match = re.search(rf"{re.escape(_FINAL_VAR_MARKER)}(.*)$", line)
                if match:
                    answer = self._parse_final_output(match.group(1))
                    if answer:
                        return FinalAnswerResult(answer), tool_calls
                continue

            # Check for FINAL marker (simple format: field=value|field=value|)
            if _FINAL_MARKER in line:
                match = re.search(rf"{re.escape(_FINAL_MARKER)}(.*)$", line)
                if match:
                    answer = self._parse_final_output(match.group(1))
                    if answer:
                        return FinalAnswerResult(answer), tool_calls
                continue

            clean_lines.append(line)

        return "\n".join(clean_lines), tool_calls

    def _execute_tool_calls(self, tool_calls: list[tuple[str, str]]) -> str:
        """Execute tool calls and return combined results."""
        results = []
        for tool_name, args_str in tool_calls:
            if tool_name not in self._tools:
                results.append(f"Error: Unknown tool '{tool_name}'")
                continue

            tool_func = self._tools[tool_name]
            try:
                # Parse arguments (simple space-split for now)
                args = args_str.split() if args_str.strip() else []
                result = tool_func(*args)
                results.append(str(result))
            except Exception as e:
                results.append(f"Error calling {tool_name}: {e}")

        return "\n".join(results)

    async def _aexecute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Async implementation of execute."""
        from just_bash.interpreter.errors import ExitError

        # Build the setup code (functions, variables)
        setup_parts = []

        # Add FINAL/FINAL_VAR functions
        setup_parts.append(self._build_final_functions())

        # Add tool functions
        if self._tools:
            setup_parts.append(self._build_tool_functions())

        # Add variable injections
        if variables:
            for name, value in variables.items():
                setup_parts.append(self._serialize_variable(name, value))

        # Execute setup code first (if any)
        if setup_parts:
            setup_code = "\n".join(setup_parts)
            try:
                await self._bash.exec(setup_code)
            except ExitError:
                pass  # Ignore exit errors in setup
            except Exception as e:
                raise CodeInterpreterError(f"Setup error: {e}") from e

        # Execute the user's code
        try:
            result = await self._bash.exec(code)
        except ExitError as e:
            # Exit with non-zero code
            raise CodeInterpreterError(f"Bash error: {e}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if "syntax" in error_msg or "parse" in error_msg:
                raise SyntaxError(f"Bash syntax error: {e}") from e
            raise CodeInterpreterError(f"Bash error: {e}") from e

        # Combine stdout and stderr
        output = result.stdout or ""
        if result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr

        # Parse the output for FINAL markers and tool calls
        parsed_result, tool_calls = self._parse_output(output)

        # Execute any tool calls
        if tool_calls:
            tool_output = self._execute_tool_calls(tool_calls)
            if isinstance(parsed_result, str) and tool_output:
                parsed_result = (parsed_result + "\n" + tool_output).strip()

        # Return the result
        if isinstance(parsed_result, FinalAnswerResult):
            return parsed_result

        if parsed_result and parsed_result.strip():
            return parsed_result.strip() + "\n"

        return None

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute bash code and return the result.

        Args:
            code: Bash code to execute
            variables: Variables to inject into the bash environment

        Returns:
            - FinalAnswerResult if FINAL() or FINAL_VAR() was called
            - str with captured stdout otherwise
            - None if no output

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On bash syntax errors
        """
        if self._shutdown:
            raise CodeInterpreterError("Interpreter has been shut down")

        if not self._started:
            self.start()

        # Handle empty code
        if not code or not code.strip():
            return None

        # Run the async execute
        return self._run_async(self._aexecute(code, variables))

    def shutdown(self) -> None:
        """Clean up resources.

        Releases the Bash interpreter instance.
        This method is idempotent - safe to call multiple times.
        """
        self._bash = None
        self._started = False
        self._shutdown = True

        # Clean up event loop if we created one
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    def __enter__(self) -> "BashInterpreter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.shutdown()


# =============================================================================
# BashRLM - RLM subclass for Bash code generation
# =============================================================================


# Bash-specific action instructions template
_BASH_ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Bash shell environment. Write Bash code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (available as shell variables, e.g., $context, $query)
- `llm_query "prompt"` - query a sub-LLM (~500K char capacity) for semantic analysis
- `echo` - ALWAYS use echo to see results
- `FINAL value` - submit final answer when done (single output)
- `FINAL value1 value2 ...` - submit final answer with multiple values
- `FINAL_VAR var_name` - submit final answer using a variable name
- Standard bash features: variables, loops, conditionals, pipes, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Use echo to understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - Pattern matching finds WHERE things are; llm_query understands WHAT things mean.

You have max {max_llm_calls} sub-LLM calls. When done, call FINAL or FINAL_VAR with your answer."""


class BashRLM:
    """Recursive Language Model for Bash code generation.

    This is a specialized version of RLM that generates Bash scripts instead of
    Python code. It uses BashInterpreter for code execution.

    Example:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4"))

        rlm = BashRLM("log_content: str -> error_count: int")
        result = rlm(log_content="error\\nerror\\ninfo\\nerror")
        print(result.error_count)
        ```

    Note:
        BashRLM requires the just-bash library: pip install just-bash
    """

    def __init__(
        self,
        signature: str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        verbose: bool = False,
        tools: dict[str, Callable[..., str]] | None = None,
        sub_lm=None,
        interpreter: BashInterpreter | None = None,
    ):
        """Initialize BashRLM.

        Args:
            signature: Defines inputs and outputs, e.g., "input_text -> word_count"
            max_iterations: Maximum shell interaction iterations
            max_llm_calls: Maximum sub-LLM calls per execution
            max_output_chars: Maximum characters from shell output
            verbose: Whether to log detailed execution info
            tools: Additional tool functions (exposed as bash functions)
            sub_lm: LM for llm_query. Defaults to dspy.settings.lm.
            interpreter: BashInterpreter instance. Creates new one if not provided.
        """
        # Import RLM here to avoid circular imports
        from dspy.predict.rlm import RLM

        # Create the underlying RLM with our custom interpreter
        self._interpreter = interpreter or BashInterpreter(tools=tools)

        self._rlm = RLM(
            signature=signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            sub_lm=sub_lm,
            interpreter=self._interpreter,
        )

        # Override the action instructions to use Bash-specific template
        self._patch_instructions()

    def _patch_instructions(self):
        """Patch the RLM's action signature to use Bash instructions."""
        import dspy
        from dspy.adapters.utils import translate_field_type
        from dspy.primitives.repl_types import REPLHistory, REPLVariable

        # Get the current signature info
        sig = self._rlm.signature
        inputs_str = ", ".join(f"${n}" for n in sig.input_fields)
        final_output_names = ", ".join(sig.output_fields.keys())
        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in sig.output_fields.items()
        )

        # Get task instructions
        task_instructions = f"{sig.instructions}\n\n" if sig.instructions else ""

        # Build new action signature with Bash instructions
        action_sig = (
            dspy.Signature({}, task_instructions + _BASH_ACTION_INSTRUCTIONS_TEMPLATE.format(
                inputs=inputs_str,
                final_output_names=final_output_names,
                output_fields=output_fields,
                max_llm_calls=self._rlm.max_llm_calls,
            ))
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the shell"), type_=list[REPLVariable])
            .append("repl_history", dspy.InputField(desc="Previous shell code executions and their outputs"), type_=REPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number"), type_=str)
            .append("reasoning", dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."), type_=str)
            .append("code", dspy.OutputField(desc="Bash code to execute."), type_=str)
        )

        # Replace the generate_action predictor
        self._rlm.generate_action = dspy.Predict(action_sig)

    def forward(self, **input_args):
        """Execute BashRLM to produce outputs."""
        return self._rlm.forward(**input_args)

    async def aforward(self, **input_args):
        """Async version of forward()."""
        return await self._rlm.aforward(**input_args)

    def __call__(self, **input_args):
        """Execute BashRLM. Alias for forward()."""
        return self.forward(**input_args)

    @property
    def signature(self):
        """The signature defining inputs and outputs."""
        return self._rlm.signature

    @property
    def tools(self):
        """User-provided tools."""
        return self._rlm.tools
