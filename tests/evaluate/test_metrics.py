# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import pytest

import dspy
from dspy.evaluate.metrics import EM, F1, HotPotF1, answer_exact_match
from dspy.predict import Predict


def test_answer_exact_match_string():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = dspy.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


@pytest.mark.parametrize("fn", [EM, F1, HotPotF1])
def test_metrics_raise_descriptive_error_on_empty_answers_list(fn):
    with pytest.raises(ValueError, match="answers_list"):
        fn("prediction", [])


def test_answer_exact_match_empty_list_answer():
    example = dspy.Example(question="q", answer=[]).with_inputs("question")
    pred = dspy.Prediction(answer="x")
    with pytest.raises(ValueError, match="answers_list"):
        answer_exact_match(example, pred)
