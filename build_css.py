#!/usr/bin/env python3
"""
Build script for Tailwind CSS
This script uses the Tailwind CLI to build the CSS file for production.
"""

import subprocess
import sys


def build_css():
    """Build the Tailwind CSS file"""
    try:
        # Check if tailwindcss is installed
        result = subprocess.run(
            ['npx', 'tailwindcss', '--version'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("Tailwind CSS CLI not found. Installing...")
            subprocess.run(['npm', 'install'], check=True)
        
        print("Building Tailwind CSS...")
        
        # Build the CSS file
        subprocess.run([
            'npx', 'tailwindcss',
            '-i', './src/input.css',
            '-o', './static/css/output.css',
            '--minify'
        ], check=True)
        
        print("✅ Tailwind CSS built successfully!")
        print("📁 Output file: static/css/output.css")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building CSS: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: npm/npx not found. Please install Node.js and npm first.")
        print("Download from: https://nodejs.org/")
        print("Download from: https://nodejs.org/")
        sys.exit(1)


if __name__ == "__main__":
    build_css() 