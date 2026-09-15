import asyncio
import subprocess
import sys
import textwrap

import dspy


def test_syncify_in_place():
    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = dspy.syncify(MyProgram())
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_with_wrapper():
    class MyProgram(dspy.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = dspy.syncify(MyProgram(), in_place=False)
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_works_with_optimizers():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question->answer")

        async def aforward(self, question: str):
            return await self.predict.acall(question=question)

    async_program = MyProgram()

    def dummy_metric(gold, pred, traces=None):
        return True

    # We only test the optimizer completes without errors, so the LM response doesn't matter.
    lm = dspy.utils.DummyLM([{"answer": "dummy"} for _ in range(100)])
    dspy.configure(lm=lm)

    dataset = [dspy.Example(question="question", answer="answer").with_inputs("question") for _ in range(10)]

    optimizer = dspy.BootstrapFewShot(metric=dummy_metric, max_bootstrapped_demos=2, max_labeled_demos=0)

    # Test syncify in place
    sync_program = dspy.syncify(async_program, in_place=True)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2

    # Test syncify with wrapper
    sync_program = dspy.syncify(async_program, in_place=False)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2



def test_run_async_inside_running_event_loop():
    # Exercises the `nest_asyncio` branch of `run_async`, taken when called from an already-running loop
    # (e.g. Jupyter). Run in a subprocess because `nest_asyncio.apply()` patches asyncio process-wide.
    script = textwrap.dedent(
        """
        import asyncio
        from dspy.utils.syncify import run_async

        async def inner():
            await asyncio.sleep(0.01)
            return 42

        async def outer():
            return run_async(inner())

        print(asyncio.run(outer()))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "42"
