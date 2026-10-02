"""`dspy.Skill`: a named block of reference material in the Agent Skills layout."""

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type
from dspy.primitives.sandbox_serializable import SandboxSerializable

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
RESOURCE_DIRS = ("scripts", "references", "assets")

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_SKILL_FILE_SUFFIXES = (".md", ".markdown", ".txt")


# The Agent Skills spec names the frontmatter key `description`, which shadows the `Type.description()`
# classmethod on instances only; `Skill.description()` still resolves to it on the class.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "description" in "Skill" shadows', category=UserWarning)

    class Skill(Type, SandboxSerializable):
        """A named block of reference material for a language model.

        A skill follows the Agent Skills layout: a directory holding a `SKILL.md` whose YAML frontmatter
        gives the skill's `name` and `description`, followed by a markdown body of instructions, with
        optional `scripts/`, `references/`, and `assets/` directories of bundled resources. A skill can
        also be loaded from a single markdown or text file, or from an inline string.

        As a signature input, a skill renders as its `SKILL.md` document: the frontmatter followed by
        the full content. As an output, a model fills `name`, `description`, and `content`, and `save()`
        writes the result as a skill directory. Inside `dspy.RLM`, a skill is injected as a sandbox variable: the model sees
        its name, description, and resource list up front and reads `.content` or a bundled resource
        from `.resources` only when it decides to.

        Example:
            ```python
            skill = dspy.Skill.load("./skills/prompt-engineering")

            class Draft(dspy.Signature):
                skill: dspy.Skill = dspy.InputField()
                task: str = dspy.InputField()
                prompt: str = dspy.OutputField()

            dspy.Predict(Draft)(skill=skill, task="Summarize support tickets.")
            ```
        """

        name: str
        content: str
        description: str | None = None
        path: Path | None = None
        frontmatter: dict[str, str] = pydantic.Field(default_factory=dict)

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
            # A model producing a Skill fills in name, description, and content only.
            schema = super().__get_pydantic_json_schema__(core_schema, handler)
            schema = handler.resolve_ref_schema(schema)
            properties = schema.get("properties")
            if isinstance(properties, dict):
                properties.pop("path", None)
                properties.pop("frontmatter", None)
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [field for field in required if field not in ("path", "frontmatter")]
            return schema

        # -- Loading and saving -------------------------------------------------------

        @classmethod
        def load(cls, source: "Skill | str | Path") -> "Skill":
            """Load a skill from a `Skill`, a path, or an inline string.

            Resolution order:

            1. A `Skill` is returned as is.
            2. A `Path` is always a path (`~` is expanded). A missing path raises `FileNotFoundError`.
            3. A `str` naming an existing file or directory (after `~` expansion) is a path.
            4. A `str` that does not exist but looks like a path raises `FileNotFoundError`. A string
               looks like a path when it contains no whitespace and either contains a path separator,
               starts with `.` or `~`, or ends with `.md`, `.markdown`, or `.txt`.
            5. Any other `str` is inline skill content. Empty content raises `ValueError`.

            A directory must contain `SKILL.md`; only that file is read, and bundled resources are listed
            by `resources()`. A file is read as UTF-8. A leading YAML frontmatter block supplies `name`
            and `description`; its other flat keys are kept in `frontmatter`; the block is stripped from
            `content`. The fallback name is the directory name, the file stem, or the first line of an
            inline skill.

            Note: a non-existent path that contains whitespace cannot be told apart from inline content
            and is loaded as inline content. Pass a `Path` for strict checking.
            """
            if isinstance(source, Skill):
                return source
            if isinstance(source, Path):
                return cls._from_path(source.expanduser())
            if not isinstance(source, str):
                raise TypeError(f"A skill must be a Skill, a str, or a Path, not {type(source).__name__}.")

            if not source.strip():
                raise ValueError("Empty skill content.")
            path = _existing_path(source)
            if path is not None:
                return cls._from_path(path)
            if _looks_like_path(source):
                raise FileNotFoundError(
                    f"Skill {source!r} looks like a path, but no such file or directory exists. "
                    "Pass the path to an existing skill file or directory, or pass inline skill content "
                    "(text containing whitespace)."
                )
            return cls._from_text(source, fallback_name=None, path=None)

        @classmethod
        def _from_path(cls, path: Path) -> "Skill":
            if not path.exists():
                raise FileNotFoundError(f"Skill path {path} does not exist.")
            if path.is_dir():
                skill_md = path / SKILL_FILE
                if not skill_md.exists():
                    raise FileNotFoundError(f"Skill directory {path} has no {SKILL_FILE}.")
                skill = cls._from_text(skill_md.read_text(encoding="utf-8"), fallback_name=path.name, path=skill_md)
                if skill.name != path.name:
                    logger.warning(
                        "Skill name %r does not match its directory name %r; the Agent Skills layout expects them "
                        "to be equal.",
                        skill.name,
                        path.name,
                    )
                return skill
            return cls._from_text(path.read_text(encoding="utf-8"), fallback_name=path.stem, path=path)

        @classmethod
        def _from_text(cls, text: str, fallback_name: str | None, path: Path | None) -> "Skill":
            text = text.strip()
            if not text:
                raise ValueError("Empty skill content.")

            meta, content = _parse_frontmatter(text)
            content = content.strip()

            if fallback_name is None:
                # The first non-empty line doubles as a display name for inline skills.
                first_line = content.splitlines()[0].lstrip("# ").strip() if content else ""
                fallback_name = ((first_line[:60] + "…") if len(first_line) > 60 else first_line) or "inline-skill"

            name = meta.pop("name", None) or fallback_name
            description = meta.pop("description", None) or None
            return cls(name=name, content=content, description=description, path=path, frontmatter=meta)

        def save(self, directory: str | Path, overwrite: bool = False) -> Path:
            """Write this skill as `<directory>/<name>/SKILL.md` and return the path of that file.

            The name must follow the Agent Skills rules: lowercase letters, digits, and single hyphens,
            at most 64 characters. Bundled resources are not copied. Raises `FileExistsError` when the
            file exists and `overwrite` is false.
            """
            validate_name(self.name)
            skill_dir = Path(directory).expanduser() / self.name
            skill_md = skill_dir / SKILL_FILE
            if skill_md.exists() and not overwrite:
                raise FileExistsError(f"{skill_md} already exists; pass overwrite=True to replace it.")

            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_md.write_text(self.format() + "\n", encoding="utf-8")
            return skill_md

        # -- Resources ----------------------------------------------------------------

        @property
        def root(self) -> Path | None:
            """The skill directory, when the skill was loaded from a `SKILL.md`."""
            if self.path is not None and self.path.name == SKILL_FILE:
                return self.path.parent
            return None

        def resources(self) -> list[str]:
            """Relative paths of the files under `scripts/`, `references/`, and `assets/`, sorted."""
            root = self.root
            if root is None:
                return []
            found = []
            for dirname in RESOURCE_DIRS:
                directory = root / dirname
                if directory.is_dir():
                    found.extend(p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file())
            return sorted(found)

        def read(self, resource: str) -> str:
            """Return the text of a bundled resource, given its path relative to the skill directory."""
            root = self.root
            if root is None:
                raise FileNotFoundError(f"Skill {self.name!r} has no skill directory, so it has no resources.")
            target = (root / resource).resolve()
            if root.resolve() not in target.parents:
                raise ValueError(f"Resource {resource!r} is outside the skill directory {root}.")
            if not target.is_file():
                raise FileNotFoundError(f"Skill {self.name!r} has no resource {resource!r}.")
            return target.read_text(encoding="utf-8")

        # -- Rendering ----------------------------------------------------------------

        def summary(self) -> str:
            """One line naming the skill: its name, description, and location."""
            text = self.name
            if self.description:
                text += f": {self.description}"
            if self.path is not None:
                text += f" ({self.path})"
            return text

        def format(self) -> str:
            """Render the skill as its `SKILL.md` document: frontmatter, a blank line, then the content.

            This is the text `save()` writes, so a rendered skill loads back as an equal skill.
            """
            lines = ["---", f"name: {self.name}"]
            if self.description:
                lines.append(f"description: {json.dumps(self.description, ensure_ascii=False)}")
            for key, value in self.frontmatter.items():
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            lines += ["---", "", self.content.strip()]
            return "\n".join(lines)

        def __str__(self) -> str:
            return self.format()

        # -- RLM sandbox ---------------------------------------------------------------

        def sandbox_setup(self) -> str:
            return "import json\nfrom types import SimpleNamespace"

        def to_sandbox(self) -> bytes:
            """Serialize the skill with its text resources; a resource that is not UTF-8 text is left out."""
            resources = {}
            for resource in self.resources():
                try:
                    resources[resource] = self.read(resource)
                except UnicodeDecodeError:
                    continue
            payload = {
                "name": self.name,
                "description": self.description,
                "content": self.content,
                "resources": resources,
            }
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
            return f"{var_name} = SimpleNamespace(**json.loads({data_expr}))"

        def rlm_preview(self, max_chars: int = 500) -> str:
            text = f"Skill {self.name!r}"
            if self.description:
                text += f": {self.description}"
            text += f"\nRead .content for the full instructions ({len(self.content):,} chars)."
            resources = self.resources()
            if resources:
                text += (
                    f"\n{len(resources)} bundled text resource(s) in .resources, a dict of path -> text: "
                    + ", ".join(resources)
                )
            return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def validate_name(name: str) -> None:
    """Raise `ValueError` unless `name` follows the Agent Skills naming rules."""
    if len(name) > _MAX_NAME_LENGTH or not _NAME_RE.match(name):
        raise ValueError(
            f"Skill name {name!r} is invalid: use lowercase letters, digits, and single hyphens, "
            f"at most {_MAX_NAME_LENGTH} characters."
        )


def _existing_path(source: str) -> Path | None:
    try:
        path = Path(source).expanduser()
        return path if path.exists() else None
    except (OSError, ValueError):  # e.g. an inline string too long to be a valid path
        return None


def _looks_like_path(source: str) -> bool:
    if re.search(r"\s", source):
        return False
    return (
        "/" in source
        or "\\" in source
        or source.startswith((".", "~"))
        or source.lower().endswith(_SKILL_FILE_SUFFIXES)
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading `--- ... ---` block into (metadata, body).

    Only flat, unindented `key: value` lines are read; nested YAML is ignored. Returns `({}, text)` when
    there is no well-formed block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            meta: dict[str, str] = {}
            for raw in lines[1:end]:
                if raw.startswith((" ", "\t")) or ":" not in raw:
                    continue
                key, _, value = raw.partition(":")
                meta[key.strip()] = _unquote(value.strip())
            return meta, "\n".join(lines[end + 1 :])
    return {}, text


def _unquote(value: str) -> str:
    """Decode a double-quoted YAML scalar as a JSON string (which `save()` writes); strip other quotes."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            decoded = json.loads(value)
            if isinstance(decoded, str):
                return decoded
        except ValueError:
            pass
    return value.strip("'\"")
