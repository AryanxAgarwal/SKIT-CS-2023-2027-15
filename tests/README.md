# Frame Ingest Validation Tests

This directory contains automated tests for validating captured frame
data and its associated metadata.

## What is validated

The frame ingest validator checks:

- Required metadata columns
- Frame filename format
- Frame ID format
- Duplicate frame IDs
- Duplicate filenames
- Timestamp format
- Frame dimensions
- Capture FPS
- Missing frame files
- Extra frame files without metadata

## Automated testing

The tests are executed using Python `unittest`.

GitHub Actions automatically runs the frame ingest test suite on
pushes and pull requests to the `main` branch.

## Test coverage

The test suite includes checks for:

1. A valid dataset
2. Missing frame files
3. Extra frame files
4. Duplicate frame IDs
