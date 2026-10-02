"""Tests for dspy.Skill: loading, saving, rendering, and RLM sandbox injection."""

import json
import logging
from pathlib import Path

import pytest

import dspy
from dspy.adapters.types.skill import validate_name
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from dspy.utils.dummies import DummyLM

SKILL_MD = """---
name: prompt-engineering
description: "How to write clear instructions"
license: MIT
version: 2
---

# Prompt engineering

Be explicit about the output format.
"""


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    root = tmp_path / "prompt-engineering"
    root.mkdir()
    (root / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (root / "references").mkdir()
    (root / "references" / "style.md").write_text("Use active voice.", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "check.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "logo.bin").write_bytes(b"\xff\xfe\x00binary")
    (root / "notes.md").write_text("not a resource", encoding="utf-8")
    return root


# --- loading ------------------------------------------------------------------


def test_load_directory_reads_skill_md_and_frontmatter(skill_dir: Path):
    skill = dspy.Skill.load(skill_dir)

    assert skill.name == "prompt-engineering"
    assert skill.description == "How to write clear instructions"
    assert skill.content == "# Prompt engineering\n\nBe explicit about the output format."
    assert skill.frontmatter == {"license": "MIT", "version": "2"}
    assert skill.path == skill_dir / "SKILL.md"
    assert skill.root == skill_dir
    assert dspy.Skill.load(str(skill_dir)) == skill
    assert dspy.Skill.load(skill) is skill


def test_load_directory_without_skill_md_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"SKILL\.md"):
        dspy.Skill.load(tmp_path)


def test_load_directory_warns_when_name_differs_from_directory(tmp_path: Path, caplog):
    root = tmp_path / "other-name"
    root.mkdir()
    (root / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        skill = dspy.Skill.load(root)
    assert skill.name == "prompt-engineering"
    assert "does not match its directory name 'other-name'" in caplog.text


def test_load_file_uses_stem_as_fallback_name(tmp_path: Path):
    path = tmp_path / "openai.md"
    path.write_text("Use markdown headers.\n", encoding="utf-8")
    skill = dspy.Skill.load(path)
    assert skill == dspy.Skill(name="openai", content="Use markdown headers.", path=path)
    assert skill.root is None
    assert skill.resources() == []


def test_load_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prompt-enginering"):
        dspy.Skill.load("./skills/prompt-enginering")
    with pytest.raises(FileNotFoundError):
        dspy.Skill.load("notes.markdown")
    with pytest.raises(FileNotFoundError):
        dspy.Skill.load(tmp_path / "missing")


def test_load_inline_content():
    skill = dspy.Skill.load("Be terse.")
    assert skill == dspy.Skill(name="Be terse.", content="Be terse.")

    long_first_line = "x" * 80 + "\nmore"
    assert dspy.Skill.load(long_first_line).name == "x" * 60 + "…"

    with_frontmatter = dspy.Skill.load("---\nname: inline\ndescription: d\n---\nBody.")
    assert (with_frontmatter.name, with_frontmatter.description, with_frontmatter.content) == ("inline", "d", "Body.")

    for empty in ("", "   \n"):
        with pytest.raises(ValueError, match="Empty skill"):
            dspy.Skill.load(empty)
    with pytest.raises(TypeError):
        dspy.Skill.load(42)


# --- resources ----------------------------------------------------------------


def test_resources_lists_files_under_the_three_resource_dirs(skill_dir: Path):
    skill = dspy.Skill.load(skill_dir)
    assert skill.resources() == ["assets/logo.bin", "references/style.md", "scripts/check.py"]
    assert skill.read("references/style.md") == "Use active voice."
    assert skill.read("scripts/check.py") == "print('ok')\n"

    with pytest.raises(FileNotFoundError, match="no resource"):
        skill.read("references/missing.md")
    with pytest.raises(ValueError, match="outside the skill directory"):
        skill.read("../notes.md")
    with pytest.raises(FileNotFoundError, match="no skill directory"):
        dspy.Skill.load("Be terse.").read("references/style.md")


# --- saving -------------------------------------------------------------------


def test_save_round_trips_through_load(tmp_path: Path):
    skill = dspy.Skill(
        name="sum-up",
        description='Summarize "briefly": one line',
        content="Write one sentence.\n\nNo preamble.",
        frontmatter={"license": "MIT"},
    )
    path = skill.save(tmp_path)
    assert path == tmp_path / "sum-up" / "SKILL.md"
    assert path.read_text(encoding="utf-8") == (
        '---\nname: sum-up\ndescription: "Summarize \\"briefly\\": one line"\nlicense: "MIT"\n---\n\n'
        "Write one sentence.\n\nNo preamble.\n"
    )

    loaded = dspy.Skill.load(path.parent)
    assert (loaded.name, loaded.description, loaded.content, loaded.frontmatter) == (
        skill.name,
        skill.description,
        skill.content,
        skill.frontmatter,
    )

    with pytest.raises(FileExistsError):
        skill.save(tmp_path)
    assert skill.save(tmp_path, overwrite=True) == path


@pytest.mark.parametrize("name", ["Prompt", "prompt engineering", "-lead", "a--b", "x" * 65, ""])
def test_save_rejects_invalid_names(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="invalid"):
        dspy.Skill(name=name, content="c").save(tmp_path)
    with pytest.raises(ValueError):
        validate_name(name)


def test_validate_name_accepts_spec_names():
    for name in ("a", "prompt-engineering", "v2-notes", "x" * 64):
        validate_name(name)


# --- rendering ----------------------------------------------------------------


def test_format_summary_and_str(skill_dir: Path):
    skill = dspy.Skill.load(skill_dir)
    block = (
        "<skill name='prompt-engineering' description='How to write clear instructions'>\n"
        "# Prompt engineering\n\nBe explicit about the output format.\n</skill>"
    )
    assert skill.format() == block
    assert str(skill) == block
    assert skill.summary() == f"prompt-engineering: How to write clear instructions ({skill_dir / 'SKILL.md'})"

    inline = dspy.Skill.load("Be terse.")
    assert inline.format() == "<skill name='Be terse.'>\nBe terse.\n</skill>"
    assert inline.summary() == "Be terse."


def test_skill_as_a_signature_input_renders_its_block():
    class Sig(dspy.Signature):
        skill: dspy.Skill = dspy.InputField()
        skills: list[dspy.Skill] = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = DummyLM([{"answer": "ok"}])
    skill = dspy.Skill(name="be-terse", content="Be terse.", description="Short answers")
    with dspy.context(lm=lm):
        assert dspy.Predict(Sig)(skill=skill, skills=[skill], question="hi").answer == "ok"

    user = lm.history[-1]["messages"][1]["content"]
    assert "[[ ## skill ## ]]\n<skill name='be-terse' description='Short answers'>\nBe terse.\n</skill>\n\n" in user
    assert "[[ ## skills ## ]]\n" in user
    assert user.count("<skill name='be-terse'") == 2
    assert "CUSTOM-TYPE" not in user


def test_skill_as_a_signature_output_parses_name_description_and_content():
    lm = DummyLM(
        [{"skill": {"name": "sum-up", "description": "One line", "content": "Do X.\nThen Y."}}],
        adapter=dspy.JSONAdapter(),
    )
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        out = dspy.Predict("notes -> skill: dspy.Skill")(notes="n")

    assert out.skill == dspy.Skill(name="sum-up", description="One line", content="Do X.\nThen Y.")

    schema = dspy.Skill.model_json_schema()
    assert set(schema["properties"]) == {"name", "description", "content"}
    assert schema["required"] == ["name", "content"]


# --- RLM sandbox ----------------------------------------------------------------


def test_skill_is_sandbox_serializable_and_reconstructs_in_plain_python(skill_dir: Path):
    skill = dspy.Skill.load(skill_dir)
    assert isinstance(skill, SandboxSerializable)

    payload = json.loads(skill.to_sandbox().decode("utf-8"))
    assert payload == {
        "name": "prompt-engineering",
        "description": "How to write clear instructions",
        "content": skill.content,
        # The binary asset is left out; text resources travel with the skill.
        "resources": {"references/style.md": "Use active voice.", "scripts/check.py": "print('ok')\n"},
    }

    namespace = {"_raw_skill": skill.to_sandbox().decode("utf-8")}
    exec(skill.sandbox_setup() + "\n" + skill.sandbox_assignment("skill", "_raw_skill"), namespace)
    injected = namespace["skill"]
    assert injected.name == "prompt-engineering"
    assert injected.content == skill.content
    assert injected.resources["references/style.md"] == "Use active voice."


def test_rlm_preview_discloses_name_description_and_resources(skill_dir: Path):
    skill = dspy.Skill.load(skill_dir)
    preview = skill.rlm_preview()
    assert preview == (
        "Skill 'prompt-engineering': How to write clear instructions\n"
        f"Read .content for the full instructions ({len(skill.content):,} chars).\n"
        "3 bundled text resource(s) in .resources, a dict of path -> text: "
        "assets/logo.bin, references/style.md, scripts/check.py"
    )
    assert skill.content not in preview
    assert skill.rlm_preview(max_chars=40).endswith("...")
    assert len(skill.rlm_preview(max_chars=40)) == 40

    variable = build_repl_variable(skill, "skill")
    assert variable.type_name == "Skill"
    assert variable.preview == preview
    assert "from types import SimpleNamespace" in variable.desc
