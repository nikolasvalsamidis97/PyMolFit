# Release Process

1. Run `python -m pytest -q` and `python -m ruff check src tests`.
2. Update `CHANGELOG.md`, `CITATION.cff`, and `src/pymolfit/_version.py`.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Install the wheel into a fresh virtual environment and import PyMolFit from
   outside the repository.
5. Commit the release, create an annotated `vX.Y.Z` tag, and push it.
6. Create the matching GitHub release only after CI passes.
7. The `Publish to PyPI` workflow publishes that exact checkout through PyPI
   Trusted Publishing.
8. Install the published version in a clean environment and run the documented
   quick start.

Configure the GitHub repository as a trusted publisher for the PyPI project
with workflow `publish.yml` and environment `pypi`. No API token is stored in
the repository or GitHub secrets.

Never reuse a published version number. If publication fails after PyPI has
accepted a file, increment the version before trying again.
