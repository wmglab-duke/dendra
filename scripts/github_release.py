"""Publish checked distributions as a GitHub Release after production PyPI upload.

This helper never creates tags or replaces assets. Run it with the validated
tag, commit and distribution artifact, using the workflow's GH_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from packaging.version import Version


def gh(*arguments: str) -> bytes:
    result = subprocess.run(["gh", *arguments], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def api(endpoint: str, *arguments: str):
    return json.loads(gh("api", endpoint, *arguments))


def verify_tag(repository: str, tag: str, commit: str) -> None:
    obj = api(f"repos/{repository}/git/ref/tags/{quote(tag, safe='')}")["object"]
    seen = set()
    while obj["type"] == "tag":
        if obj["sha"] in seen:
            raise ValueError("Remote tag contains a cycle.")
        seen.add(obj["sha"])
        obj = api(f"repos/{repository}/git/tags/{obj['sha']}")["object"]
    if obj["type"] != "commit" or obj["sha"] != commit:
        raise ValueError("Remote tag does not point to the validated commit.")


def changelog_notes(text: str, tag: str, repository: str) -> str:
    """Select exactly one version section and make relative links portable."""
    version = re.escape(tag[1:])
    headings = list(re.finditer(r"(?m)^##[ \t]+([^\n]+)$", text))
    matches = [
        index
        for index, heading in enumerate(headings)
        if re.fullmatch(rf"(?:v?{version}|\[v?{version}\])(?:[ \t]+.*)?", heading[1])
    ]
    if len(matches) != 1:
        raise ValueError(f"Changelog must contain one exact section for {tag}.")
    index = matches[0]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
    body = text[headings[index].end() : end].strip()
    if not body:
        raise ValueError(f"Changelog section for {tag} is empty.")

    def link(match):
        label, destination = match.groups()
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or destination.startswith(("#", "/")):
            return match[0]
        destination = urlunsplit(
            (
                "https",
                "github.com",
                f"/{repository}/blob/{quote(tag, safe='')}/{quote(parsed.path, safe='/%')}",
                parsed.query,
                parsed.fragment,
            )
        )
        return f"[{label}]({destination})"

    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^\s)]+)\)", link, body) + "\n"


def pages(endpoint: str):
    return [item for page in api(endpoint, "--paginate", "--slurp") for item in page]


def find_release(repository: str, tag: str):
    # The listing includes drafts for the write-scoped job token; by-tag
    # lookup is documented only for published releases.
    matches = [
        release
        for release in pages(f"repos/{repository}/releases?per_page=100")
        if release["tag_name"] == tag
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple GitHub releases exist for {tag}.")
    return matches[0] if matches else None


def distribution_files(directory: Path):
    files = {}
    for path in sorted(directory.iterdir()):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.name.endswith((".whl", ".tar.gz"))
            or "#" in path.name
            or path.stat().st_size == 0
        ):
            raise ValueError(f"Unexpected or empty distribution artifact: {path.name}")
        files[path.name] = path
    if not files:
        raise ValueError("Checked distribution directory is empty.")
    return files


def verify_assets(repository: str, release, files):
    assets = pages(f"repos/{repository}/releases/{release['id']}/assets?per_page=100")
    present = {}
    for asset in assets:
        name = asset["name"]
        if name in present or name not in files:
            raise ValueError(f"Unexpected or duplicate release asset: {name}")
        expected = hashlib.sha256(files[name].read_bytes()).hexdigest()
        if asset["state"] != "uploaded" or asset["size"] != files[name].stat().st_size:
            raise ValueError(f"Incomplete or different release asset: {name}")
        digest = asset.get("digest")
        if digest is None:
            digest = (
                "sha256:"
                + hashlib.sha256(
                    gh(
                        "api",
                        f"repos/{repository}/releases/assets/{asset['id']}",
                        "--header",
                        "Accept: application/octet-stream",
                    )
                ).hexdigest()
            )
        if digest != "sha256:" + expected:
            raise ValueError(f"Release asset bytes differ: {name}")
        present[name] = asset
    return present


def release(repository: str, tag: str, commit: str, dist_dir: Path, changelog=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must be OWNER/REPOSITORY.")
    if not tag.startswith("v"):
        raise ValueError("Release tag must start with v.")
    version = Version(tag[1:])
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("Validated commit must be a full lowercase Git SHA.")
    files = distribution_files(dist_dir)
    notes = (
        changelog_notes(Path(changelog).read_text(encoding="utf-8"), tag, repository)
        if changelog is not None
        else None
    )
    verify_tag(repository, tag, commit)
    current = find_release(repository, tag)
    if current is None:
        command = [
            "release",
            "create",
            tag,
            "--repo",
            repository,
            "--verify-tag",
            "--target",
            commit,
            "--draft",
            "--title",
            tag,
        ]
        if version.is_prerelease:
            command.append("--prerelease")
        with tempfile.TemporaryDirectory() as temporary:
            if notes is None:
                command.append("--generate-notes")
            else:
                path = Path(temporary) / "notes.md"
                path.write_text(notes, encoding="utf-8")
                command.extend(("--notes-file", str(path)))
            gh(*command)
        current = find_release(repository, tag)
        if current is None:
            raise ValueError("Created draft could not be found.")
    if current["prerelease"] != version.is_prerelease:
        raise ValueError("Existing release has a different prerelease status.")
    if notes is not None and (current.get("body") or "").strip() != notes.strip():
        raise ValueError("Existing release notes differ from the tagged changelog.")
    present = verify_assets(repository, current, files)
    missing = sorted(files.keys() - present.keys())
    if not current["draft"]:
        if missing:
            raise ValueError("Published release is missing checked distributions.")
        print(f"Verified existing release {tag}; nothing changed.")
        return
    for name in missing:
        gh("release", "upload", tag, str(files[name]), "--repo", repository)
    if set(verify_assets(repository, current, files)) != set(files):
        raise ValueError("Draft does not contain every checked distribution.")
    verify_tag(repository, tag, commit)
    gh("release", "edit", tag, "--repo", repository, "--draft=false", "--verify-tag")
    print(f"Published verified GitHub release {tag}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--changelog", type=Path)
    args = parser.parse_args()
    try:
        release(args.repository, args.tag, args.commit, args.dist_dir, args.changelog)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"GitHub Release failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
