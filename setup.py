#!/usr/bin/env python3
"""
Setup Script for Smart Feedback Summarizer
Automates installation and verification
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False

def main():
    print_header("SMART FEEDBACK SUMMARIZER - SETUP")
    
    # Step 1: Check Python version
    print("🔍 Step 1: Checking Python version...")
    if not check_python_version():
        print("\n⚠️  Please install Python 3.8 or higher")
        return
    
    # Step 2: Create virtual environment
    print("\n🔍 Step 2: Setting up virtual environment...")
    if not Path("venv").exists():
        if run_command(
            f"{sys.executable} -m venv venv",
            "Creating virtual environment"
        ):
            print("✅ Virtual environment created")
        else:
            print("⚠️  Continuing without virtual environment...")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine pip command
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Step 3: Install requirements
    print("\n🔍 Step 3: Installing dependencies...")
    print("⚠️  This may take 5-10 minutes and download ~2GB of models")
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if run_command(
        f"{pip_cmd} install -r requirements.txt",
        "Installing dependencies"
    ):
        print("✅ All dependencies installed")
    else:
        print("❌ Failed to install dependencies")
        print("Try manually: pip install -r requirements.txt")
        return
    
    # Step 4: Download spaCy model
    print("\n🔍 Step 4: Downloading spaCy model...")
    run_command(
        f"{python_cmd} -m spacy download en_core_web_sm",
        "Downloading spaCy model"
    )
    
    # Step 5: Generate sample data
    print("\n🔍 Step 5: Generating sample dataset...")
    if run_command(
        f"{python_cmd} generate_sample_data.py",
        "Generating sample data"
    ):
        print("✅ Sample data generated")
    
    # Step 6: Verify installation
    print("\n🔍 Step 6: Verifying installation...")
    print("\nChecking key dependencies:")
    
    dependencies = [
        "pandas",
        "numpy",
        "pyspark",
        "transformers",
        "bertopic",
        "streamlit",
        "plotly"
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            result = subprocess.run(
                f"{python_cmd} -c \"import {dep}\"",
                shell=True,
                check=True,
                capture_output=True
            )
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ {dep}")
            all_ok = False
    
    # Final summary
    print_header("SETUP COMPLETE")
    
    if all_ok:
        print("✅ All systems ready!")
        print("\n📚 Next Steps:")
        print("\n1. Run the complete pipeline:")
        if os.name == 'nt':
            print("   venv\\Scripts\\python main.py")
        else:
            print("   venv/bin/python main.py")
        
        print("\n2. Launch the dashboard:")
        if os.name == 'nt':
            print("   venv\\Scripts\\streamlit run dashboard.py")
        else:
            print("   venv/bin/streamlit run dashboard.py")
        
        print("\n3. View documentation:")
        print("   Open README.md for detailed instructions")
        print("   Open DOCUMENTATION.md for technical details")
        
        print("\n📊 Sample data location:")
        print(f"   {Path('data/customer_feedback.csv').absolute()}")
        
    else:
        print("⚠️  Some dependencies failed to install.")
        print("Please check the error messages above and try manual installation.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
