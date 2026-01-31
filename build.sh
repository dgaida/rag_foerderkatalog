#!/usr/bin/env bash
# build.sh - Render Build Script

set -o errexit  # Exit on error

echo "🔨 Starting Render Build..."

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

# Download data if needed
echo "📥 Checking for required data files..."
if [ ! -f "input/foerderkatalog_export.csv" ] || [ ! -f "data/vector_hf.index" ]; then
    echo "⚠️  Data files missing - attempting download..."
    python download_data.py
else
    echo "✅ Data files present"
fi

echo "✅ Build completed successfully!"
