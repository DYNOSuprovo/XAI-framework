#!/usr/bin/env python
"""Create GitHub issues from the drafts in docs/issues/.

Each draft is a Markdown file with a small front matter block::

    ---
    title: Add Explanation.to_markdown()
    labels: [enhancement, good first issue]
    ---
    body...

Usage::

    export GITHUB_TOKEN=ghp_...          # needs "issues: write" on the repo
    python scripts/create_issues.py --repo <owner>/XAI-Framework --dry-run
    python scripts/create_issues.py --repo <owner>/XAI-Framework

Only the standard library is used. Labels that do not exist yet are created first.
Drafts whose title already exists as an open issue are skipped, so re-running is safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
DRAFT_DIR = Path(__file__).resolve().parent.parent / "docs" / "issues"

LABEL_COLOURS = {
    "bug": "d73a4a",
    "enhancement": "a2eeef",
    "documentation": "0075ca",
    "good first issue": "7057ff",
    "help wanted": "008672",
    "explainer": "fbca04",
    "testing": "bfd4f2",
    "performance": "e99695",
    "research": "c5def5",
    "infrastructure": "d4c5f9",
    "text": "f9d0c4",
    "image": "f9d0c4",
    "deep-learning": "f9d0c4",
}

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_draft(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta: dict = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key.strip() == "labels":
            meta["labels"] = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
        else:
            meta[key.strip()] = value
    meta["body"] = text[match.end() :].strip() + "\n"
    if "title" not in meta:
        raise ValueError(f"{path.name}: front matter needs a title")
    return meta


class GitHub:
    def __init__(self, token: str, repo: str) -> None:
        self.token = token
        self.repo = repo

    def request(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{API}{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or b"null")

    def existing_labels(self) -> set[str]:
        return {
            lbl["name"] for lbl in self.request("GET", f"/repos/{self.repo}/labels?per_page=100")
        }

    def ensure_label(self, name: str) -> None:
        payload = {"name": name, "color": LABEL_COLOURS.get(name, "ededed")}
        try:
            self.request("POST", f"/repos/{self.repo}/labels", payload)
        except urllib.error.HTTPError as e:
            if e.code != 422:  # 422 = already exists
                raise

    def open_issue_titles(self) -> set[str]:
        titles: set[str] = set()
        page = 1
        while True:
            items = self.request(
                "GET", f"/repos/{self.repo}/issues?state=open&per_page=100&page={page}"
            )
            if not items:
                return titles
            titles.update(i["title"] for i in items if "pull_request" not in i)
            page += 1

    def create_issue(self, title: str, body: str, labels: list[str]) -> str:
        issue = self.request(
            "POST", f"/repos/{self.repo}/issues", {"title": title, "body": body, "labels": labels}
        )
        return issue["html_url"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--dry-run", action="store_true", help="print what would be created")
    parser.add_argument("--only", nargs="*", help="draft file names to create (default: all)")
    args = parser.parse_args(argv)

    drafts = sorted(DRAFT_DIR.glob("*.md"))
    if args.only:
        drafts = [d for d in drafts if d.name in args.only or d.stem in args.only]
    if not drafts:
        print("no drafts found", file=sys.stderr)
        return 1

    parsed = [(d, parse_draft(d)) for d in drafts]
    if args.dry_run:
        for path, meta in parsed:
            print(f"{path.name}: {meta['title']}  [{', '.join(meta.get('labels', []))}]")
        print(f"\n{len(parsed)} issues would be created in {args.repo}")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("set GITHUB_TOKEN (a token with issues: write)", file=sys.stderr)
        return 2
    gh = GitHub(token, args.repo)

    have_labels = gh.existing_labels()
    for _, meta in parsed:
        for label in meta.get("labels", []):
            if label not in have_labels:
                gh.ensure_label(label)
                have_labels.add(label)

    existing = gh.open_issue_titles()
    created = 0
    for path, meta in parsed:
        if meta["title"] in existing:
            print(f"skip   {path.name} (already open)")
            continue
        url = gh.create_issue(meta["title"], meta["body"], meta.get("labels", []))
        print(f"create {path.name} -> {url}")
        created += 1
    print(f"\n{created} issues created, {len(parsed) - created} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
