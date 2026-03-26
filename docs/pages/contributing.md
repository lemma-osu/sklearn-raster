## Setup

This project is developed with [uv](https://docs.astral.sh/uv/) and [poe](https://poethepoet.natn.io/index.html). Install the project, development tooling, and pre-commit hooks with:

```bash
uv sync
uv run poe install
```

## Linting and Formatting

Run linting, formatting, and type checking with:

```bash
uv run poe check
```

## Tests

Add tests to the `tests` dir. Run pytest with:

```bash
uv run poe test
uv run poe coverage
```

To test against different Python versions, use:

```bash
uv run --python 3.14 poe test
```

## Docs

Write new documentation in the `docs/pages` directory. Add them to the `nav` in `docs/mkdocs.yml`. Serve documentation with:

```bash
uv run poe docs
```

## Releasing

First, use `uv` to update the version number in a new release branch and merge into `main`.

```bash
$ uv version --bump [major|minor|patch|alpha|beta|rc|post|dev]
```

Checkout `main` and confirm that it is up-to-date with the remote, including the bumped version. Finally, create and push the release tag.

```bash
$ git checkout main
$ git pull
$ git tag "$(uv version --short)"
$ git push --tags
```

Pushing the updated tag will trigger [a workflow](https://github.com/lemma-osu/sklearn-raster/actions/workflows/publish.yml) that publishes the release to PyPI.