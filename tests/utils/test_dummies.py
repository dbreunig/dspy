import dspy
from dspy.utils.dummies import DummyLM


def test_dummy_lm_follow_examples_uses_matching_demo():
    dspy.configure(lm=DummyLM([{"answer": "red"}], follow_examples=True))
    predictor = dspy.Predict("question -> answer")
    predictor.demos = [dspy.Example(question="What color is the sky?", answer="blue").with_inputs("question")]

    assert predictor(question="What color is the sky?").answer == "blue"


def test_dummy_lm_follow_examples_falls_back_to_answers_without_matching_demo():
    dspy.configure(lm=DummyLM([{"answer": "blue"}], follow_examples=True))
    predictor = dspy.Predict("question -> answer")

    assert predictor(question="What color is the sky?").answer == "blue"
