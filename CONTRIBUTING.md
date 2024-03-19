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

## Publish

Increment versions with Hatch:

```bash
hatch version <patch|minor|major>
```

Build and publish with Hatch:

```bash
hatch clean
hatch build
hatch publish
```