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

## Confirm the distribution name

Check the `dendra` name separately on TestPyPI and PyPI before the first public
`main` promotion. The promoted README and documentation advertise
`pip install dendra`, so the final distribution name must already be settled.
A yanked release, a renamed project, or deletion of every release does not by
itself make a PyPI project name available. If
<https://pypi.org/project/dendra/> still belongs to the previous unrelated
project, complete an owner transfer and confirm that a Dendra maintainer can
open its settings page. If the name is genuinely absent on an index, use the
pending-publisher procedure below.

Do not make the GitHub repository public, publish its documentation, or upload
a distribution until this check passes. If a transfer cannot be completed,
choose the final distribution name first and update package metadata,
installation commands, trusted publisher settings, and the `dendra-models`
dependency together.

Before the first public promotion, remove the temporary pre-release notices in
`README.md`, `docs/index.rst`, and `docs/installation.md` so the published
package presents PyPI as the current installation path. At that point, enable
the public documentation link and any GitHub or PyPI status badges. Until then,
keep shared README assets and links repository relative so the canonical GitLab
project remains usable while the GitHub mirror is private.

## Configure the public GitHub repository

After the initial `main` promotion described below, make
`wmglab-duke/dendra` public and confirm that `main` is its default branch.
Enable Issues and pull requests and allow repository forks. Keep the
publishing workflow in GitLab so every promoted revision carries the same
checks.

In **Settings → Pages**, set the source to **GitHub Actions**. The
`Documentation` workflow validates the Sphinx build on pull requests and
publishes `main` to <https://wmglab-duke.github.io/dendra/>.
The first `main` promotion can reach the documentation deployment before Pages
has been enabled and fail there with a 404. After selecting GitHub Actions as
the Pages source, rerun that workflow on `main`; its Sphinx build remains the
same validation gate.

Add the GitHub repository as a second remote in the canonical GitLab checkout
if it is not already configured:

```sh
git remote add github git@github.com:wmglab-duke/dendra.git
```

Pull requests from forks run with read-only repository permissions. The
publishing jobs are restricted to `wmglab-duke/dendra`, require either an
explicit TestPyPI dispatch or a selected `v*` tag, and receive an OpenID Connect
token only after all distribution checks pass.

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

## Configure GitHub publishing

Keep the publishing workflow in GitLab at
`.github/workflows/publish.yml`, so promoted source revisions carry the same
configuration. In the GitHub repository, enable Actions and create two
environments: `testpypi` and `pypi`. Restrict `testpypi` to
`release-candidate/*` branches and `pypi` to `v*` tags. Add a required reviewer
to `pypi` so a tag cannot publish until a maintainer approves the deployment.
Maintainers select public releases by choosing which existing GitLab tags to
push to GitHub. Production publication, including manual dispatch, requires
selecting a version tag.

Register separate GitHub Trusted Publishers on
[TestPyPI](https://test.pypi.org/manage/account/publishing/) and
[PyPI](https://pypi.org/manage/account/publishing/):

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | `dendra` | `dendra` |
| Repository owner | `wmglab-duke` | `wmglab-duke` |
| Repository name | `dendra` | `dendra` |
| Workflow filename | `publish.yml` | `publish.yml` |
| Environment | `testpypi` | `pypi` |

If a project does not yet exist on an index, register a pending publisher on
that index's account Publishing page. For a project the team already controls,
add the publisher through the project's Publishing page. A pending publisher
does **not** reserve a name: the project is created by its first successful
upload. TestPyPI and PyPI use separate accounts and registrations. This setup
uses short-lived GitHub identity credentials and needs no PyPI API token in
repository secrets.

## Promote source without publishing

Start from a clean checkout and fetch the current GitLab `main`. Push the
freshly fetched remote branch directly, so a stale local `main` cannot select
the wrong revision:

```sh
git fetch origin --tags
git push --no-follow-tags github refs/remotes/origin/main:refs/heads/main
```

This runs GitHub build and test checks without uploading to either package
index. Set GitHub's default branch to `main` after its initial push so the
workflow's manual dispatch is available. The explicit `--no-follow-tags`
prevents a personal Git configuration from sending annotated release tags.
Keep development branches and internal version tags on GitLab unless you
deliberately select them for promotion. Avoid `--tags`, `--follow-tags`, and
`--mirror` when pushing to GitHub.

## Rehearse a selected release on TestPyPI

Choose an existing, tested GitLab version tag. The selected tag must contain
`publish.yml`; a tag created before the publishing workflow was added cannot
run it. Select a later release containing the workflow instead of moving an
existing tag.

Replace `v0.27.0` in the commands below with the chosen tag. Push its exact
commit to a candidate branch, without sending the version tag:

```sh
git fetch origin --tags
git push --no-follow-tags github \
  'v0.27.0^{commit}:refs/heads/release-candidate/0.27.0'
```

In GitHub Actions, open the publishing workflow, select **Run workflow**, and
choose `release-candidate/0.27.0`. Use `publish_target=none` to validate first,
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
  '/path/to/checked-dists/dendra-0.27.0-py3-none-any.whl[solvers]'
python -m pip install --index-url https://test.pypi.org/simple/ \
  --force-reinstall --no-deps --no-cache-dir --only-binary=:all: 'dendra==0.27.0'
python -m pip check
python -I - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import dendra
import dendra.models.analysis as analysis

for module in (dendra, analysis):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert version("dendra") == "0.27.0"
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
git push --no-follow-tags github refs/tags/v0.27.0:refs/tags/v0.27.0
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
  --no-cache-dir --only-binary=:all: 'dendra[solvers]==0.27.0'
python -m pip check
```

Run the isolated import/version check above again. Uploaded distribution
filenames cannot be replaced; fixes to a published package require a new
version. Preserve Dendra's custom Duke research license in source and
distribution metadata; publishing through GitHub or PyPI does not change its
terms.
