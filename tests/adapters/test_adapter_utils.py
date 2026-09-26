# ruff: noqa: UP007

from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel

from dspy.adapters.utils import _type_adapter_for, parse_value, serialize_for_json


class Profile(BaseModel):
    name: str
    age: int


def test_parse_value_str_annotation():
    # Test basic string conversion
    assert parse_value(123, str) == "123"
    assert parse_value(True, str) == "True"
    assert parse_value("hello", str) == "hello"
    assert parse_value(None, str) == "None"
    assert parse_value([1, 2, 3], str) == "[1, 2, 3]"


def test_parse_value_pydantic_types():
    # Test with pydantic BaseModel - JSON string input
    json_str = '{"name": "John", "age": 30}'
    result = parse_value(json_str, Profile)
    assert isinstance(result, Profile)
    assert result.name == "John"
    assert result.age == 30

    # Test with pydantic BaseModel - dict input
    dict_input = {"name": "Jane", "age": 25}
    result = parse_value(dict_input, Profile)
    assert isinstance(result, Profile)
    assert result.name == "Jane"
    assert result.age == 25

    # Test with invalid pydantic data
    with pytest.raises(Exception):
        parse_value('{"name": "John"}', Profile)  # missing required age field


def test_parse_value_basic_types():
    # Test int
    assert parse_value("42", int) == 42
    assert parse_value(42, int) == 42

    # Test float
    assert parse_value("3.14", float) == 3.14
    assert parse_value(3.14, float) == 3.14

    # Test bool
    assert parse_value("true", bool) is True
    assert parse_value(True, bool) is True
    assert parse_value("false", bool) is False

    # Test list
    assert parse_value("[1, 2, 3]", list[int]) == [1, 2, 3]
    assert parse_value([1, 2, 3], list[int]) == [1, 2, 3]


def test_parse_value_literal():
    # Test Literal type
    assert parse_value("option1", Literal["option1", "option2"]) == "option1"
    assert parse_value("option2", Literal["option1", "option2"]) == "option2"

    # Test Literal with quotes and prefixes
    assert parse_value("'option1'", Literal["option1", "option2"]) == "option1"
    assert parse_value('"option1"', Literal["option1", "option2"]) == "option1"
    assert parse_value("Literal[option1]", Literal["option1", "option2"]) == "option1"
    assert parse_value("str[option1]", Literal["option1", "option2"]) == "option1"

    # Test invalid literal
    with pytest.raises(ValueError):
        parse_value("invalid", Literal["option1", "option2"])


def test_parse_value_literal_non_string_members():
    # Text adapters yield literal values as strings; coerce them to the member type.
    assert parse_value("2", Literal[1, 2, 3]) == 2
    assert parse_value('"2"', Literal[1, 2, 3]) == 2
    assert parse_value("True", Literal[True, False]) is True

    with pytest.raises(ValueError):
        parse_value("42", Literal[1, 2, 3])


def test_parse_value_union():
    # Test Union with None (Optional)
    assert parse_value("test", Optional[str]) == "test"
    assert parse_value("test", str | None) == "test"
    assert parse_value("5", int | None) == 5
    assert parse_value(None, Optional[str]) is None
    assert parse_value("text with [placeholder]", Optional[str]) == "text with [placeholder]"
    assert parse_value("text with [placeholder]", str | None) == "text with [placeholder]"

    # Test Union fallback to str
    assert parse_value("fallback", Union[int, str, None]) == "fallback"
    assert parse_value(5, Union[int, str, None]) == 5
    assert parse_value("fallback", int | str | None) == "fallback"
    assert parse_value(5, int | str | None) == 5
    assert parse_value("text with [placeholder]", Union[int, str, None]) == "text with [placeholder]"


def test_parse_value_json_repair():
    # Test cases where json_repair is needed
    assert parse_value('{"key": "value"}', dict) == {"key": "value"}

    # Test ast.literal_eval fallback
    assert parse_value("{'key': 'value'}", dict) == {"key": "value"}

    # Test fallback to original value when parsing fails
    malformed = "not json or literal"
    with pytest.raises(Exception):
        parse_value(malformed, dict)


class _Unserializable:
    def __str__(self):
        return "unserializable"


@pytest.mark.parametrize(
    "value",
    [
        "hello",
        3,
        2.5,
        True,
        None,
        [1, "a", {"b": 2}],
        {"k": [1, 2]},
        Profile(name="John", age=30),
        _Unserializable(),
    ],
)
def test_serialize_for_json_matches_uncached_type_adapter(value):
    from pydantic import TypeAdapter

    try:
        expected = TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        expected = str(value)
    assert serialize_for_json(value) == expected


def test_serialize_for_json_reuses_type_adapter_per_type():
    # One adapter per type: the same object serves every value of that type.
    assert _type_adapter_for(Profile) is _type_adapter_for(Profile)
    assert _type_adapter_for(str) is not _type_adapter_for(int)

    hits_before = _type_adapter_for.cache_info().hits
    serialize_for_json("first")
    serialize_for_json("second")
    assert _type_adapter_for.cache_info().hits >= hits_before + 1
