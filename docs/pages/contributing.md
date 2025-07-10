## Setup

Install pre-commit hooks to auto-run linting and formatting:

```bash
hatch run pre-commit install
```

## Linting and Formatting

Run linting and formatting manually with Hatch:

```bash
hatch run pre-commit run --all-files
```

## Tests

Add tests to the `tests` dir. Run pytest via the Hatch `test` environment scripts:

```bash
hatch run test:all
hatch run test:cov
```

## Docs

Write new documentation in the `docs/pages` directory. Add them to the `nav` in `docs/mkdocs.yml`. Build and serve mkdocs documentation via the Hatch `docs` environment scripts:

```bash
hatch run docs:serve
hatch run docs:build
```

## Releasing

First, use `hatch` to [update the version number](https://hatch.pypa.io/latest/version/#updating) in a new release branch and merge into `main`.

```bash
$ hatch version [major|minor|patch|alpha|beta|rc|post|dev]
```

Checkout `main` and confirm that it is up-to-date with the remote, including the bumped version. Finally, create and push the release tag.

```bash
$ git checkout main
$ git pull
$ git tag "$(hatch version)"
$ git push --tags
```

Pushing the updated tag will trigger [a workflow](https://github.com/lemma-osu/sklearn-raster/actions/workflows/publish.yml) that publishes the release to PyPI.