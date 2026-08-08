"""The declaration, as a file in the customer's own repository.

`.berth/classes.yaml` in their repo, not a JSON file handed to us.

That placement is the whole design. The declaration is what the agent is
allowed to watch, which paths it may write, and what bound each class holds.
Putting it in their repository means changing it is a pull request they review,
the history of what was declared and when is in their git log, and revoking us
is deleting a file rather than emailing.

It also means the thing that constrains the agent and the thing the agent
edits live in the same place under the same review, which is the property that
makes the first integration approvable.

Format is deliberately small. Every field is something the customer already
knows; nothing here requires them to understand the estimator.

    version: 1
    repo:
      allowed_paths:
        - deploy/voice.yaml
        - deploy/embed.yaml
    classes:
      - name: voice-agent-prod
        model_id: NousResearch/Meta-Llama-3-8B
        model: llama3-8b
        running_on: h100-pcie
        config_path: deploy/voice.yaml
        slo:
          metric: p99_ttft_ms
          bound_ms: 800
        workload:
          concurrency: 8
          prompt_tokens: 512
          output_tokens: 128
"""

from __future__ import annotations

from dataclasses import dataclass, field

from berth.agent import WatchedClass
from berth.github import RepoTarget

CONFIG_PATH = ".berth/classes.yaml"


class DeclarationError(ValueError):
    """The declaration is malformed or claims something it may not.

    Raised loudly rather than defaulted around. A declaration is the document
    that bounds what the agent may do, and quietly filling in a missing field
    would mean the agent operating under terms nobody wrote down.
    """


@dataclass
class Declaration:
    """One parsed `.berth/classes.yaml`."""

    version: int
    allowed_paths: tuple[str, ...]
    classes: list[WatchedClass] = field(default_factory=list)
    default_branch: str = "main"

    def repo_target(self, owner: str, name: str) -> RepoTarget:
        return RepoTarget(owner=owner, name=name,
                          default_branch=self.default_branch,
                          allowed_paths=self.allowed_paths)


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise DeclarationError(
            f"{where} is missing {key!r}. Every field in a declaration is "
            f"something you already know, and filling one in for you would "
            f"mean the agent operating under terms nobody wrote down.")
    return d[key]


def parse(raw: dict, repo: str = "") -> Declaration:
    """Validate a loaded declaration.

    Takes a dict rather than a path so the loader can be YAML, JSON, or a
    fixture, and so this is testable without a filesystem.
    """
    version = raw.get("version")
    if version != 1:
        raise DeclarationError(
            f"version must be 1, got {version!r}. An unversioned declaration "
            f"cannot be safely reinterpreted when the format changes.")

    repo_block = raw.get("repo") or {}
    allowed = tuple(repo_block.get("allowed_paths") or ())
    if not allowed:
        raise DeclarationError(
            "repo.allowed_paths is empty. The agent writes only to paths you "
            "name, and an empty list means none. This is deliberate: an agent "
            "that can write anywhere is one nobody grants access to.")

    owner, _, name = repo.partition("/")
    classes = []
    for i, c in enumerate(raw.get("classes") or []):
        where = f"classes[{i}]"
        # Checked in the order they appear in the file, so the error names the
        # first thing missing as a reader scans down rather than the first
        # thing the parser happened to want.
        for required in ("name", "model_id", "model", "running_on",
                         "config_path", "slo", "workload"):
            _require(c, required, where)
        cfg = c["config_path"]
        if cfg not in allowed:
            raise DeclarationError(
                f"{where} names config_path {cfg!r}, which is not in "
                f"repo.allowed_paths. Add it there deliberately rather than "
                f"having the agent infer permission from use.")
        slo = _require(c, "slo", where)
        wl = _require(c, "workload", where)
        classes.append(WatchedClass(
            workload_class=_require(c, "name", where),
            model_id=_require(c, "model_id", where),
            model_key=_require(c, "model", where),
            current_silicon=_require(c, "running_on", where),
            current_model_version="",     # filled by the first watch
            slo_metric=slo.get("metric", "p99_ttft_ms"),
            slo_bound_ms=float(_require(slo, "bound_ms", f"{where}.slo")),
            batch=int(_require(wl, "concurrency", f"{where}.workload")),
            prompt_tokens=int(_require(wl, "prompt_tokens", f"{where}.workload")),
            output_tokens=int(_require(wl, "output_tokens", f"{where}.workload")),
            config_path=cfg,
            repo=repo))

    if not classes:
        raise DeclarationError(
            "no classes declared. A declaration with nothing to watch is not "
            "a configuration error, but it is almost certainly a mistake, so "
            "it is refused rather than silently doing nothing.")

    names = [c.workload_class for c in classes]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise DeclarationError(
            f"duplicate class names {sorted(dupes)}. A class name is the key "
            f"the agent tracks state against, so two of the same would make "
            f"proposals and their outcomes ambiguous.")

    return Declaration(version=version, allowed_paths=allowed,
                       classes=classes,
                       default_branch=repo_block.get("default_branch", "main"))


def load_yaml(text: str, repo: str = "") -> Declaration:
    """Parse YAML text. Falls back to a minimal reader if PyYAML is absent,
    because the core has no dependencies and a customer should not need to
    install one to be read."""
    try:
        import yaml
        raw = yaml.safe_load(text)
    except ImportError:
        raw = _minimal_yaml(text)
    if not isinstance(raw, dict):
        raise DeclarationError("declaration must be a mapping at the top level")
    return parse(raw, repo)


def _minimal_yaml(text: str):
    """Enough YAML for this format and no more.

    Deliberately not a general parser. It handles the nesting, lists and
    scalars this file uses, and anything else raises rather than being
    silently misread, because a declaration misread is an agent operating
    under terms nobody wrote.
    """
    import json
    import re

    def scalar(v):
        v = v.strip()
        if v.startswith(("'", '"')) and v[-1] == v[0]:
            return v[1:-1]
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
        if v in ("true", "false"):
            return v == "true"
        return v

    root: dict = {}
    stack = [(-1, root)]
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise DeclarationError(f"line {lineno}: indentation is inconsistent")
        parent = stack[-1][1]

        if body.startswith("- "):
            item = body[2:].strip()
            if not isinstance(parent, list):
                raise DeclarationError(
                    f"line {lineno}: list item outside a list")
            if ":" in item and not item.endswith(":"):
                k, _, v = item.partition(":")
                d = {k.strip(): scalar(v)}
                parent.append(d)
                stack.append((indent, d))
            elif item.endswith(":"):
                d = {}
                parent.append(d)
                stack.append((indent, d))
                stack.append((indent + 1, d))
            else:
                parent.append(scalar(item))
            continue

        if body.endswith(":"):
            key = body[:-1].strip()
            child: dict | list = {}
            # Peek: a list follows if the next non-blank line is a dash.
            rest = text.splitlines()[lineno:]
            for nxt in rest:
                if not nxt.strip() or nxt.lstrip().startswith("#"):
                    continue
                if nxt.lstrip().startswith("- "):
                    child = []
                break
            parent[key] = child
            stack.append((indent, child))
            continue

        k, _, v = body.partition(":")
        if not isinstance(parent, dict):
            raise DeclarationError(f"line {lineno}: mapping inside a list item")
        parent[k.strip()] = scalar(v)
    return json.loads(json.dumps(root))
