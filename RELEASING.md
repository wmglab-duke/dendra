# Maintaining and publishing Dendra

This guide is for project maintainers. Dendra is developed on
[GitLab](https://gitlab.oit.duke.edu/dendra-dev/dendra), which remains the
canonical source and the `origin` remote. Selected versions are promoted to the
public-facing
[`wmglab-duke/dendra`](https://github.com/wmglab-duke/dendra) repository for
external contributions and for GitHub Actions to build, test, and publish.
Internal GitLab versions can be skipped when choosing public releases.

GitLab CI updates the version, changelog, and version tag after the required
checks pass on `main`. GitHub validates promoted source and publishes selected
releases; it does not bump versions or edit the changelog.

Keep the GitHub workflows in canonical GitLab history so every promoted
revision carries the configuration that GitHub executes. Public releases are
selected by promoting immutable GitLab tags; uploads remain restricted to an
explicit TestPyPI rehearsal or a selected `v*` production tag.

## Configure a maintainer checkout

Each checkout used for GitHub contribution or release work needs a `github`
remote pointing to the public mirror. Add it once if it is absent:

```sh
git remote add github git@github.com:wmglab-duke/dendra.git
```

## Integrate an external contribution

Review the GitHub pull request and let its distribution checks finish. When the
change is ready for the required internal CI suite, fetch its exact commits
and merge them into a new branch based on the current GitLab `develop` branch.
Start from a clean canonical checkout; stop if `git status --short` prints any
paths. Replace `123` and the example commit message below:

```sh
git status --short
git fetch origin develop
git fetch github \
  +refs/pull/123/head:refs/remotes/github/pull/123
git rev-parse refs/remotes/github/pull/123
git diff origin/develop...refs/remotes/github/pull/123
git switch --create contrib/github-123 origin/develop
git merge --no-ff refs/remotes/github/pull/123 \
  --message "feat(scope): integrate GitHub PR 123" \
  --message "https://github.com/wmglab-duke/dendra/pull/123"
git push --set-upstream origin contrib/github-123
```

Inspect untrusted changes before running them locally. Open a GitLab merge
request from `contrib/github-123` into `develop` and run the full internal CI
pipeline. Leave the GitHub pull request open while this happens. Keep its exact
commits in the history: do not squash, rebase, or cherry-pick the internal
merge request. Resolve any conflicts in the merge commit rather than rewriting
the contributor's commits. For an accepted change, do not use GitHub's Merge or
Close button. After the change reaches GitLab `main` and that branch is
synchronized to GitHub, the contributor's pull-request head is an ancestor of
public `main`, preserving authorship and allowing GitHub to mark the pull
request as merged. Preserve those commits when merging `develop` into `main`
as well; a squash or rebase at that stage breaks the ancestry GitHub uses.

## Promote source without publishing

Start from a clean checkout and fetch the current GitLab `main`. Push the
freshly fetched remote branch directly, so a stale local `main` cannot select
the wrong revision:

```sh
git fetch origin --tags
git push --no-follow-tags github refs/remotes/origin/main:refs/heads/main
```

This runs GitHub build and test checks without uploading to either package
index. The explicit `--no-follow-tags` prevents a personal Git configuration
from sending annotated release tags. Keep development branches and internal
version tags on GitLab unless you deliberately select them for promotion.
Avoid `--tags`, `--follow-tags`, and `--mirror` when pushing to GitHub.
If the promoted revision changes `docs/dendra-models-revision.txt`, first make
sure that exact commit exists in the public `wmglab-duke/dendra-models` mirror.

## Rehearse a selected release on TestPyPI

Choose an existing, tested GitLab version tag. The selected tag must contain
`publish.yml`; a tag created before the publishing workflow was added cannot
run it. Select a later release containing the workflow instead of moving an
existing tag.

Replace `vX.Y.Z` and `X.Y.Z` below with the chosen tag and version. Push its
exact commit to a candidate branch, without sending the version tag:

```sh
git fetch origin --tags
git push --no-follow-tags github \
  'vX.Y.Z^{commit}:refs/heads/release-candidate/X.Y.Z'
```

In GitHub Actions, open the publishing workflow, select **Run workflow**, and
choose `release-candidate/X.Y.Z`. Use `publish_target=none` to validate first,
then `publish_target=testpypi` to build, test, and upload to TestPyPI. The
production publishing job is skipped during this rehearsal. Confirm that
TestPyPI contains the selected version's wheel and source archive.

Verify a fresh installation. Download the successful workflow's
`checked-dists` artifact and use its wheel to install dependencies from the
normal package index first. For example, replace the wheel path and version
with those from the chosen run:

```sh
DENDRA_INSTALL_CHECK="$(mktemp -d)"
python -m venv "$DENDRA_INSTALL_CHECK/venv"
source "$DENDRA_INSTALL_CHECK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple/ \
  '/path/to/checked-dists/dendra-X.Y.Z-py3-none-any.whl[solvers]'
python -m pip install --index-url https://test.pypi.org/simple/ \
  --force-reinstall --no-deps --no-cache-dir --only-binary=:all: 'dendra==X.Y.Z'
python -m pip check
python -I - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import dendra
import dendra.models.analysis as analysis

for module in (dendra, analysis):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert version("dendra") == "X.Y.Z"
print(version("dendra"), dendra.__file__, analysis.__file__)
PY
```

This replaces the artifact installation with the TestPyPI wheel while keeping
dependency downloads on PyPI. Avoid using TestPyPI as an extra index: the
selected source of the Dendra distribution should be explicit. Run these
checks in isolated mode so the local source checkout cannot masquerade as the
installed release.

## Publish the selected version to PyPI

After the rehearsal and fresh-install check pass, send only the selected
existing version tag:

```sh
git push --no-follow-tags github refs/tags/vX.Y.Z:refs/tags/vX.Y.Z
```

**This tag push starts production publication.** GitHub rebuilds and tests
that exact revision before its PyPI job uploads the checked wheel and source
archive. GitHub's `main` does not need to move to publish the selected tag.
Manual dispatch with `publish_target=pypi` is also a production publication
action; select the intended `v*` release tag. A production dispatch from a
branch fails before building.

Verify the version on [PyPI](https://pypi.org/project/dendra/) and install it in
a fresh environment with:

```sh
python -m pip install --index-url https://pypi.org/simple/ \
  --no-cache-dir --only-binary=:all: 'dendra[solvers]==X.Y.Z'
python -m pip check
```

Run the isolated import/version check above again. Uploaded distribution
filenames cannot be replaced; fixes to a published package require a new
version. Preserve Dendra's custom Duke research license in source and
distribution metadata; publishing through GitHub or PyPI does not change its
terms.
