# Contributing

Changes are proposed through GitHub pull requests and reviewed before they are
merged. External contributors cannot modify the protected upstream branch
directly; they work in a fork or feature branch.

Set up a development environment with:

```bash
python -m pip install -e ".[dev,plot]"
python -m ruff check src tests
python -m pytest -q
```

Scientific-result changes need focused unit tests, a real-spectrum regression
case when redistributable data exist, and a changelog entry explaining the
expected effect. Do not add instrument-, target-, or spectrum-specific fitted
coefficients to package defaults.

Large observations, downloaded catalogues, generated products, and local
benchmark outputs belong outside Git. Add only compact redistributable
fixtures needed for deterministic tests.
