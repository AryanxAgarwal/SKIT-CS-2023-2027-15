# AI-Based Driver Drowsiness and Behavior Monitoring System

## Project Overview

This project focuses on AI-based monitoring of driver drowsiness and
driving behavior using captured frame data and extracted features.

The system includes data recording, feature extraction, labeling,
validation, and automated testing components.

## Repository Components

The repository contains modules related to:

- Frame and data recording
- Raw feature extraction
- Feature labeling
- Dataset management
- Frame ingest validation
- Automated testing
- Project configuration
- Automated reporting

## Frame Ingest Validation

The frame ingest validator checks the consistency and validity of
captured frames and their metadata.

The validation includes:

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

## Automated Testing

Automated tests are implemented using Python `unittest`.

The test suite currently covers:

1. Valid dataset validation
2. Missing frame detection
3. Extra frame detection
4. Duplicate frame ID detection

## Continuous Integration

GitHub Actions is configured to run the Python test suite automatically
for pushes and pull requests targeting the `main` branch.

The CI workflow uses Python 3.11 and executes the frame ingest
validation tests.

## Project Structure

```text
.
├── tests/
│   ├── __init__.py
│   ├── frame_ingest_validator.py
│   ├── test_frame_ingest_validator.py
│   └── README.md
├── config.py
├── extract_features.py
├── generate_report.py
├── label_clips.py
├── landmarks.py
└── README.md
