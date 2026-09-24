#!/usr/bin/env bash
# setup_env.sh
# Sprint 1 - installs everything needed to run camera_test.py,
# stability_test.py, and save_frames_metadata.py on the target board.

set -e

echo "[setup_env] Updating apt package list..."
sudo apt-get update

echo "[setup_env] Installing system dependencies for OpenCV..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libjpeg-dev \
    v4l-utils

echo "[setup_env] Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "[setup_env] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup_env] Done. Activate the environment with: source scripts/.venv/bin/activate"
echo "[setup_env] Then run: python3 camera_test.py --config ../configs/config.yaml"
