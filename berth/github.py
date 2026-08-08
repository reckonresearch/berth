"""Opening the pull request.

The agent produces a proposal. This puts it in the customer's repository.

**Scope, deliberately narrow.** Read a file, write a branch, open a pull
request. It cannot merge, cannot push to a default branch, and cannot touch a
path the customer did not declare. Those are enforced here rather than left to
a token scope, because a token scope is a promise about configuration and this
is a promise about code.

**Why a pull request and not an API call into their deploy system.** A pull
request is reviewable, discussable, amendable, rejectable and revertible, and
it leaves the decision in the customer's own history. An API call into a
deployment leaves it in ours.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class RefusedByPolicy(GitHubError):
    """The operation is outside what this integration is allowed to do.

    Distinct from a transport failure on purpose: a policy refusal is a bug in
    the caller and should never be retried, while a transport failure often
    should.
    """


@dataclass
class RepoTarget:
    """One repository and the paths the agent may touch in it."""

    owner: str
    name: str
    default_branch: str = "main"
    allowed_paths: tuple[str, ...] = ()

    def check(self, path: str):
        """A path outside the declaration is refused here, not at review.

        An agent that can write anywhere is an agent nobody grants access to,
        and the narrowest possible permission is what makes the first
        integration approvable.
        """
        if not self.allowed_paths:
            raise RefusedByPolicy(
                "no allowed_paths declared. This integration writes only to "
                "paths the customer named, and an empty list means none.")
        if path not in self.allowed_paths:
            raise RefusedByPolicy(
                f"{path!r} is not among the declared paths "
                f"{list(self.allowed_paths)}. Widening this is a conversation "
                f"with the customer, not a code change.")


def _request(method: str, url: str, token: str, body: dict | None = None,
             opener=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "berth-agent/0.1 (reckonresearch.com)",
        "Content-Type": "application/json",
    })
    try:
        with (opener or urllib.request.urlopen)(req, timeout=30) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise GitHubError(f"{method} {url} -> {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise GitHubError(f"{method} {url}: {e}") from e


class GitHubClient:
    """The smallest surface that can open a pull request."""

    def __init__(self, token: str, *, request=_request):
        if not token:
            raise GitHubError("a token is required")
        self._token = token
        self._request = request

    def _call(self, method, path, body=None):
        return self._request(method, f"{API}{path}", self._token, body)

    # -- reads ------------------------------------------------------------

    def head_sha(self, repo: RepoTarget) -> str:
        ref = self._call("GET", f"/repos/{repo.owner}/{repo.name}"
                                f"/git/ref/heads/{repo.default_branch}")
        return ref["object"]["sha"]

    def read_file(self, repo: RepoTarget, path: str, ref: str | None = None):
        repo.check(path)
        q = f"?ref={ref}" if ref else ""
        r = self._call("GET", f"/repos/{repo.owner}/{repo.name}"
                              f"/contents/{path}{q}")
        return base64.b64decode(r["content"]).decode(), r["sha"]

    # -- writes -----------------------------------------------------------

    def create_branch(self, repo: RepoTarget, branch: str, from_sha: str):
        if branch == repo.default_branch:
            raise RefusedByPolicy(
                f"refusing to write to {branch!r}, which is the default "
                f"branch. This integration opens pull requests; it does not "
                f"commit to trunk.")
        return self._call("POST", f"/repos/{repo.owner}/{repo.name}/git/refs",
                          {"ref": f"refs/heads/{branch}", "sha": from_sha})

    def put_file(self, repo: RepoTarget, path: str, content: str, *,
                 branch: str, message: str, sha: str | None = None):
        repo.check(path)
        if branch == repo.default_branch:
            raise RefusedByPolicy("refusing to commit to the default branch")
        body = {"message": message, "branch": branch,
                "content": base64.b64encode(content.encode()).decode()}
        if sha:
            body["sha"] = sha
        return self._call("PUT", f"/repos/{repo.owner}/{repo.name}"
                                 f"/contents/{path}", body)

    def open_pr(self, repo: RepoTarget, *, branch: str, title: str, body: str):
        return self._call("POST", f"/repos/{repo.owner}/{repo.name}/pulls",
                          {"title": title, "body": body, "head": branch,
                           "base": repo.default_branch})

    def merge(self, *_a, **_kw):
        """Never. Present so the refusal is explicit rather than an absence."""
        raise RefusedByPolicy(
            "this integration does not merge. The agent proposes and the "
            "customer disposes, and that boundary is in the contract as well "
            "as here.")


# ------------------------------------------------------------------ apply

def apply_edit(original: str, old_line: str, new_line: str) -> str:
    """Replace exactly one line, or refuse.

    A configuration edit that matches nothing has gone stale, and one that
    matches twice is ambiguous. Both are refused rather than guessed, because
    an agent that guesses at a deployment file is an agent that gets its
    access revoked once.
    """
    hits = [ln for ln in original.splitlines() if ln.strip() == old_line.strip()]
    if len(hits) != 1:
        raise RefusedByPolicy(
            f"expected exactly one line matching {old_line.strip()!r}, found "
            f"{len(hits)}. The configuration has changed since the estimate, "
            f"so the diff is stale and re-estimating is the correct response.")
    out, done = [], False
    for ln in original.splitlines():
        if not done and ln.strip() == old_line.strip():
            out.append(ln.replace(old_line.strip(), new_line.strip()))
            done = True
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if original.endswith("\n") else "")


def open_proposal(client: GitHubClient, repo: RepoTarget, proposal, *,
                  branch_prefix: str = "berth/placement"):
    """Put one proposal in the repository as a pull request.

    Returns the API response. Raises RefusedByPolicy before touching anything
    if the proposal would write outside the declared paths.
    """
    path = proposal.watched.config_path
    repo.check(path)

    d = proposal.decision
    old_line = f"accelerator: {d.incumbent}"
    new_line = f"accelerator: {d.recommended}"

    content, sha = client.read_file(repo, path)
    updated = apply_edit(content, old_line, new_line)

    branch = (f"{branch_prefix}/{proposal.watched.workload_class}"
              f"-{d.recommended}").replace("_", "-")
    client.create_branch(repo, branch, client.head_sha(repo))
    client.put_file(repo, path, updated, branch=branch, sha=sha,
                    message=proposal.title)
    return client.open_pr(repo, branch=branch, title=proposal.title,
                          body=proposal.body)
