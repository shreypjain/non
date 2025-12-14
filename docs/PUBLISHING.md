# Publishing to PyPI

This document explains how to publish new versions of the nons package to PyPI.

## Prerequisites

- PyPI account with API token stored in `.pypirc`
- `uv` installed for building and publishing
- All tests passing
- Updated version number in `pyproject.toml` and `nons/__init__.py`

## Quick Publish

For quick publishing using stored credentials:

```bash
# Publish to TestPyPI (recommended for testing)
./scripts/publish.sh testpypi

# Publish to production PyPI
./scripts/publish.sh pypi
```

## Manual Publishing Process

### Step 1: Update Version

Update the version number in two places:

1. `pyproject.toml` - Update the `version` field
2. `nons/__init__.py` - Update the `__version__` variable

### Step 2: Update Changelog

Document changes in the CHANGELOG.md or release notes.

### Step 3: Build the Package

```bash
uv build
```

This creates distribution files in the `dist/` directory:
- `nons-X.Y.Z-py3-none-any.whl` (wheel)
- `nons-X.Y.Z.tar.gz` (source distribution)

### Step 4: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
uv publish --token $(grep -A 2 "\[testpypi\]" .pypirc | grep password | cut -d'=' -f2 | tr -d ' ') --publish-url https://test.pypi.org/legacy/

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nons==X.Y.Z
```

### Step 5: Publish to PyPI

```bash
# Upload to PyPI
uv publish --token $(grep -A 2 "\[pypi\]" .pypirc | grep password | cut -d'=' -f2 | tr -d ' ')
```

### Step 6: Verify Installation

```bash
pip install nons==X.Y.Z
python -c "import nons; print(nons.__version__)"
```

### Step 7: Tag the Release

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

## Credentials Management

PyPI credentials are stored in `.pypirc` (not committed to git):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

## Links

- Production Package: https://pypi.org/project/nons/
- Test Package: https://test.pypi.org/project/nons/
- PyPI Token Management: https://pypi.org/manage/account/token/
- TestPyPI Token Management: https://test.pypi.org/manage/account/token/

## Troubleshooting

### File Already Exists Error

If you get a "File already exists" error, you cannot overwrite an existing version. You must:
1. Increment the version number
2. Build and publish the new version

### Authentication Failed

If authentication fails:
1. Verify your token in `.pypirc` is correct
2. Regenerate the token on PyPI if needed
3. Update `.pypirc` with the new token

### Missing Dependencies

If build or publish fails due to missing dependencies:
```bash
uv sync --dev
```
