# Contributing to Dendra

Thank you for helping improve Dendra. Bug reports, feature ideas,
documentation, tests, and code changes are all useful contributions.

## Before you start

Dendra is developed internally on GitLab. After the external repository opens,
users can report issues and submit pull requests through
[`wmglab-duke/dendra`](https://github.com/wmglab-duke/dendra) without GitLab
access. Until then, internal contributors should continue to use their GitLab
checkout and submit merge requests there. The environment, testing,
documentation, and commit guidance below applies on both hosts.

For a bug report, include:

- a short, reproducible example;
- the behavior you expected and what happened instead;
- your operating system and Python, PyTorch, and Dendra versions; and
- how you installed Dendra.

For a feature request, describe the workflow you want to support and why the
existing API does not cover it. Opening an issue before a large change helps
confirm the design and scope before you invest substantial time.

## Set up a development environment

Internal contributors can use their existing GitLab checkout. After the GitHub
repository opens, external contributors should fork `wmglab-duke/dendra`, then
clone their fork. Replace `YOUR-ACCOUNT` with your GitHub username or
organization:

```sh
git clone https://github.com/YOUR-ACCOUNT/dendra.git
cd dendra
git remote add upstream https://github.com/wmglab-duke/dendra.git
```

Create or activate a Python 3.11 or newer environment, then install Dendra in
editable mode and enable the repository hooks:

On a Windows host, perform this setup inside WSL2; the required NEURON package
does not publish native Windows wheels on PyPI. See the
[installation guide](docs/installation.md) for platform details.

```sh
python -m pip install --editable ".[dev,solvers]"
pre-commit install
pre-commit install --hook-type commit-msg
```

The `solvers` extra supplies the recommended native CPU solvers used by the
full test suite. If no compatible solver package is available for your
platform, use `python -m pip install --editable ".[dev]"`; unbranched CPU
cables can use Dendra's PyTorch fallback.

Create a focused branch from the integration branch used by your repository
host. Internal GitLab changes start from `origin/develop`:

```sh
git fetch origin
git switch --create fix/short-description origin/develop
```

External GitHub contributions start from `upstream/main`:

```sh
git fetch upstream
git switch --create fix/short-description upstream/main
```

## Make a change

Keep each pull or merge request focused on one problem. Add or update tests for
changed behavior, and update docstrings or user documentation when an interface
or workflow changes. New public APIs should use the conventions already present
in nearby modules.

Formatting and static checks are configured through `pre-commit`. Run them
across your changed files before submitting:

```sh
pre-commit run --all-files
```

Use a [Conventional Commit](https://www.conventionalcommits.org/) message in
the form `type(optional-scope): summary` so the automated release process can
classify the change. For example:

```text
fix(analysis): retain events inside the requested window
feat(models): add a trainable membrane parameter
docs: clarify CPU solver installation
```

`cz commit` can guide you through this format. Do not edit the package version
or changelog manually; continuous integration handles both after changes reach
`main`.

## Test the change

Run the smallest relevant tests while developing, then run the required CPU
and non-CUDA NEURON test lane before opening a pull or merge request:

```sh
python -m pytest tests -W error -m "cpu or (neuron and not cuda)"
```

If your change affects CUDA behavior, run the relevant CUDA tests on a machine
with a supported GPU:

```sh
python -m pytest tests -W error -m cuda
```

For documentation changes, install the documentation dependencies and require
a warning-free build that executes every notebook. The notebooks use models
from the companion `dendra-models` repository. Install the revision recorded
for the documentation into the same environment:

```sh
python -m pip install --editable ".[doc,solvers]"
DENDRA_MODELS_REVISION="$(cat docs/dendra-models-revision.txt)"
python -m pip install --no-deps \
  "dendra-models @ git+https://github.com/wmglab-duke/dendra-models.git@${DENDRA_MODELS_REVISION}"
sphinx-build -E -W --keep-going \
  -D nb_execution_mode=force \
  -D nb_execution_allow_errors=0 \
  -D nb_execution_raise_on_error=1 \
  -b html docs docs/_build/html
```

Full notebook execution also requires the `ffmpeg` executable on `PATH` for
the multicontact animation tutorial. Install it with your operating system's
package manager before running the build.

The [testing section of the README](README.md#-testing-and-code-coverage)
explains the test markers, coverage checks, and CUDA sanitizer lane in more
detail.

## Submit the change

Internal contributors should push their branch to GitLab and open a merge
request into `develop`:

```sh
git push --set-upstream origin fix/short-description
```

After the GitHub repository opens, external contributors should push their
branch to their fork and open a pull request against `wmglab-duke/dendra`'s
`main` branch:

```sh
git push --set-upstream origin fix/short-description
```

In the pull- or merge-request description, explain:

- the problem and the approach you took;
- any user-visible or compatibility implications;
- the tests you ran; and
- any follow-up work that is intentionally outside the change.

Keep the branch current if the repository host reports conflicts, and address
review and CI feedback with additional commits. Leave the request open during
review. GitHub checks build and exercise distributions for external pull
requests, and maintainers run the additional internal validation. Accepted
external commits reach public `main` with their authorship preserved.

By contributing, you agree that your contribution may be distributed under
the terms in [LICENSE.md](LICENSE.md).
