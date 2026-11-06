"""Offline release lifecycle checks; no package imports or network access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[1] / "github_release.py"
SPEC = importlib.util.spec_from_file_location("github_release_under_test", HELPER)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)

COMMIT = "a" * 40
REPOSITORY = "example/project"


class GitHub:
    """Model draft/asset/publication state behind the actual CLI boundary."""

    def __init__(self, tag="v0.5.1"):
        self.tag = tag
        self.commit = COMMIT
        self.current = None
        self.assets = []
        self.contents = {}
        self.calls = []
        self.fail_upload = None
        self.retag_after_upload = False
        self.annotated = True

    @property
    def mutations(self):
        return [call for call in self.calls if call[0] == "release"]

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "api":
            endpoint = args[1]
            if "/git/ref/tags/" in endpoint:
                obj = (
                    {"type": "tag", "sha": "b" * 40}
                    if self.annotated
                    else {"type": "commit", "sha": self.commit}
                )
                return json.dumps({"object": obj}).encode()
            if "/git/tags/" in endpoint:
                return json.dumps(
                    {"object": {"type": "commit", "sha": self.commit}}
                ).encode()
            if "/releases/assets/" in endpoint:
                self.assert_binary_header(args)
                return self.contents[int(endpoint.rsplit("/", 1)[1])]
            if "/assets?" in endpoint:
                return json.dumps([self.assets]).encode()
            if endpoint.endswith("/releases?per_page=100"):
                return json.dumps([[self.current] if self.current else []]).encode()
            raise AssertionError(f"Unexpected API endpoint: {endpoint}")
        assert args[0] == "release" and "--repo" in args
        assert args[args.index("--repo") + 1] == REPOSITORY
        assert args[2] == self.tag
        if args[1] == "create":
            assert self.current is None
            assert "--verify-tag" in args and "--draft" in args
            assert args[args.index("--target") + 1] == COMMIT
            body = (
                Path(args[args.index("--notes-file") + 1]).read_text()
                if "--notes-file" in args
                else "GitHub-generated version notes."
            )
            self.current = {
                "id": 10,
                "tag_name": self.tag,
                "draft": True,
                "prerelease": "--prerelease" in args,
                "body": body,
            }
        elif args[1] == "upload":
            assert "--clobber" not in args
            path = Path(args[3])
            if self.fail_upload == path.name:
                raise RuntimeError("Interrupted upload")
            self.add_asset(path.name, path.read_bytes())
            if self.retag_after_upload:
                self.commit = "c" * 40
        elif args[1] == "edit":
            assert "--draft=false" in args and "--verify-tag" in args
            self.current["draft"] = False
        else:
            raise AssertionError(f"Unexpected release command: {args}")
        return b""

    @staticmethod
    def assert_binary_header(args):
        assert "Accept: application/octet-stream" in args

    def add_asset(self, name, content, *, digest=True):
        identifier = len(self.assets) + 1
        self.assets.append(
            {
                "id": identifier,
                "name": name,
                "state": "uploaded",
                "size": len(content),
                "digest": (
                    "sha256:" + hashlib.sha256(content).hexdigest() if digest else None
                ),
            }
        )
        self.contents[identifier] = content


class GitHubReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.wheel = self.dist / "project-0.5.1-py3-none-any.whl"
        self.wheel.write_bytes(b"wheel")
        self.sdist = self.dist / "project-0.5.1.tar.gz"
        self.sdist.write_bytes(b"sdist")
        self.changelog = self.root / "CHANGELOG.md"
        self.changelog.write_text(
            "# Changelog\n\n## v0.5.1 (2026-11-06)\n\n### Fix\n\n"
            "- Correct the solver.\n\n## v0.5.0\n\nOld version notes.\n"
        )
        self.github = GitHub()
        self.patcher = patch.object(release, "gh", self.github)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def run_release(self, *, changelog=True):
        release.release(
            REPOSITORY,
            self.github.tag,
            COMMIT,
            self.dist,
            self.changelog if changelog else None,
        )

    def test_create_verified_draft_upload_exact_assets_then_publish(self):
        self.run_release()
        self.assertFalse(self.github.current["draft"])
        self.assertEqual(
            {asset["name"] for asset in self.github.assets},
            {self.wheel.name, self.sdist.name},
        )
        self.assertEqual(
            self.github.current["body"], "### Fix\n\n- Correct the solver.\n"
        )
        self.assertEqual(
            [call[1] for call in self.github.mutations],
            ["create", "upload", "upload", "edit"],
        )

    def test_interrupted_upload_resumes_draft_without_replacing_existing_asset(self):
        self.github.fail_upload = self.sdist.name
        with self.assertRaisesRegex(RuntimeError, "Interrupted"):
            self.run_release()
        self.assertTrue(self.github.current["draft"])
        self.assertEqual(len(self.github.assets), 1)
        self.github.fail_upload = None
        self.run_release()
        self.assertFalse(self.github.current["draft"])
        self.assertEqual(sum(call[1] == "create" for call in self.github.mutations), 1)
        self.assertEqual(
            sum(
                call[1] == "upload" and call[3] == str(self.wheel)
                for call in self.github.mutations
            ),
            1,
        )

    def test_published_rerun_does_not_mutate_release_or_assets(self):
        self.run_release()
        before = deepcopy((self.github.current, self.github.assets))
        writes = len(self.github.mutations)
        self.run_release()
        self.assertEqual((self.github.current, self.github.assets), before)
        self.assertEqual(len(self.github.mutations), writes)

    def test_digestless_asset_uses_downloaded_bytes(self):
        self.run_release()
        self.github.assets[0]["digest"] = None
        self.run_release()
        self.assertTrue(
            any(
                "Accept: application/octet-stream" in call for call in self.github.calls
            )
        )
        self.github.contents[1] = b"other"
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self.run_release()

    def test_mismatched_asset_is_never_replaced_in_draft_or_published_release(self):
        self.run_release()
        self.wheel.write_bytes(b"other")  # Same size, different SHA256.
        for draft in (False, True):
            with self.subTest(draft=draft):
                self.github.current["draft"] = draft
                writes = len(self.github.mutations)
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    self.run_release()
                self.assertEqual(len(self.github.mutations), writes)

    def test_published_missing_asset_is_not_repaired_by_mutation(self):
        self.run_release()
        self.github.assets.pop()
        writes = len(self.github.mutations)
        with self.assertRaisesRegex(ValueError, "missing checked"):
            self.run_release()
        self.assertEqual(len(self.github.mutations), writes)

    def test_unexpected_and_duplicate_assets_fail_without_mutation(self):
        self.run_release()
        for extra in (dict(self.github.assets[0]), {"name": "unreviewed.zip"}):
            with self.subTest(extra=extra):
                self.github.assets.append(extra)
                writes = len(self.github.mutations)
                with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
                    self.run_release()
                self.assertEqual(len(self.github.mutations), writes)
                self.github.assets.pop()

    def test_wrong_remote_commit_prevents_any_release_creation(self):
        self.github.commit = "c" * 40
        with self.assertRaisesRegex(ValueError, "validated commit"):
            self.run_release()
        self.assertEqual(self.github.mutations, [])

    def test_incomplete_asset_is_not_accepted_as_a_successful_upload(self):
        self.run_release()
        original = deepcopy(self.github.assets[0])
        for key, value in (("state", "starter"), ("size", original["size"] + 1)):
            with self.subTest(key=key):
                self.github.assets[0] = {**original, key: value}
                writes = len(self.github.mutations)
                with self.assertRaisesRegex(ValueError, "Incomplete or different"):
                    self.run_release()
                self.assertEqual(len(self.github.mutations), writes)

    def test_existing_notes_or_prerelease_status_are_not_silently_overwritten(self):
        self.run_release()
        original = deepcopy(self.github.current)
        for key, value in (("body", "Different notes."), ("prerelease", True)):
            with self.subTest(key=key):
                self.github.current = {**original, key: value}
                writes = len(self.github.mutations)
                with self.assertRaises(ValueError):
                    self.run_release()
                self.assertEqual(len(self.github.mutations), writes)

    def test_retag_during_upload_prevents_publication(self):
        self.github.retag_after_upload = True
        with self.assertRaisesRegex(ValueError, "validated commit"):
            self.run_release()
        self.assertTrue(self.github.current["draft"])
        self.assertNotIn("edit", [call[1] for call in self.github.mutations])

    def test_lightweight_tag_and_generated_prerelease_notes(self):
        self.github.annotated = False
        self.github.tag = "v0.6.0rc1"
        self.run_release(changelog=False)
        creation = next(call for call in self.github.mutations if call[1] == "create")
        self.assertIn("--generate-notes", creation)
        self.assertIn("--prerelease", creation)
        self.assertTrue(self.github.current["prerelease"])

    def test_missing_empty_or_duplicate_changelog_section_fails_before_api(self):
        for text in (
            "## v0.5.10\nWrong version.\n",
            "## v0.5.1\n\n",
            "## v0.5.1\nA\n## v0.5.1\nB\n",
        ):
            with self.subTest(text=text):
                self.changelog.write_text(text)
                with self.assertRaises(ValueError):
                    self.run_release()
                self.assertEqual(self.github.calls, [])

    def test_changelog_relative_links_target_tag_without_changing_external_or_anchor(
        self,
    ):
        body = release.changelog_notes(
            "## v0.5.1 (unreleased)\n"
            "[upgrade](docs/upgrading.rst#inputs) [web](https://example.org/x) "
            "[local](#section) [space](docs/file%20name.md)\n\n## v0.5.0\nOld.\n",
            "v0.5.1",
            REPOSITORY,
        )
        self.assertIn(
            "[upgrade](https://github.com/example/project/blob/v0.5.1/docs/upgrading.rst#inputs)",
            body,
        )
        self.assertIn("[web](https://example.org/x)", body)
        self.assertIn("[local](#section)", body)
        self.assertIn(
            "[space](https://github.com/example/project/blob/v0.5.1/docs/file%20name.md)",
            body,
        )
        self.assertNotIn("Old.", body)

    def test_invalid_or_empty_artifact_directory_fails_before_api(self):
        for name, data in (("unexpected.txt", b"text"), ("empty.whl", b"")):
            with self.subTest(name=name):
                path = self.dist / name
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.run_release()
                path.unlink()
        self.wheel.unlink()
        self.sdist.unlink()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.run_release()
        self.assertEqual(self.github.calls, [])


if __name__ == "__main__":
    unittest.main()
