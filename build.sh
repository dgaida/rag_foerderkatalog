#!/usr/bin/env bash
# build.sh - Render Build Script

set -o errexit  # Exit on error

echo "🔨 Starting Render Build..."

# Upgrade pip
pip install --upgrade pip

# Install minimal dependencies for download script
echo "📦 Installing download dependencies..."
pip install requests tqdm

# Download data FIRST
echo "📥 Downloading required data files..."
python download_data.py

# Install full dependencies
echo "📦 Installing full dependencies..."
pip install -e .

echo "✅ Build completed successfully!"
