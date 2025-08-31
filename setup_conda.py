#!/usr/bin/env python3
"""
Conda Setup Script for AI-Enhanced LiDAR Classification System
Automated setup with conda environment management
"""

import subprocess
import sys
import os
from pathlib import Path
import json


def run_command(cmd, check=True, shell=False):
    """Run command with proper error handling"""
    print(f"🔧 Running: {cmd}")
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()

        result = subprocess.run(cmd, capture_output=True, text=True, check=check, shell=shell)

        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠️ Warning: {result.stderr}")

        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        print(f"📝 Output: {e.stdout}")
        print(f"📝 Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_conda_installation():
    """Check if conda is installed and accessible"""
    try:
        result = run_command("conda --version", check=False)
        if result.returncode == 0:
            print(f"✅ Conda found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Conda not found in PATH")
            return False
    except FileNotFoundError:
        print("❌ Conda not installed or not in PATH")
        return False


def detect_cuda_version():
    """Detect CUDA version for optimal PyTorch installation"""
    try:
        # Try nvidia-smi first
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0 and "CUDA Version" in result.stdout:
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version: ")[1].split()[0]
                    major, minor = cuda_version.split('.')[:2]

                    if int(major) >= 12:
                        return "12.1"
                    elif int(major) == 11 and int(minor) >= 8:
                        return "11.8"
                    else:
                        return "11.8"  # Default fallback

        # Fallback to nvcc if available
        result = run_command("nvcc --version", check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "release" in line:
                    version = line.split("release ")[1].split(",")[0]
                    if version.startswith("12"):
                        return "12.1"
                    else:
                        return "11.8"

        print("⚠️ CUDA not detected, using CPU-only PyTorch")
        return "cpu"

    except:
        print("⚠️ Could not detect CUDA version, defaulting to 11.8")
        return "11.8"


def create_conda_environment():
    """Create conda environment with optimal settings"""
    env_name = "ai-enhanced-lidar"

    print(f"🐍 Creating conda environment: {env_name}")

    # Check if environment already exists
    result = run_command(f"conda info --envs", check=False)
    if env_name in result.stdout:
        print(f"⚠️ Environment {env_name} already exists")
        response = input("🔄 Remove existing environment and recreate? (y/n): ").lower()
        if response == 'y':
            print(f"🗑️ Removing existing environment...")
            run_command(f"conda env remove -n {env_name} -y")
        else:
            print("✅ Using existing environment")
            return env_name

    # Create environment from environment.yml
    if Path("environment.yml").exists():
        print("📋 Creating environment from environment.yml...")
        run_command(f"conda env create -f environment.yml")
    else:
        print("📋 Creating environment with basic packages...")
        # Detect CUDA version
        cuda_version = detect_cuda_version()

        if cuda_version == "cpu":
            pytorch_cmd = f"conda create -n {env_name} python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y"
        else:
            pytorch_cmd = f"conda create -n {env_name} python=3.10 pytorch torchvision torchaudio pytorch-cuda={cuda_version} -c pytorch -c nvidia -y"

        run_command(pytorch_cmd)

    print(f"✅ Conda environment '{env_name}' created successfully")
    return env_name


def install_pip_dependencies(env_name):
    """Install pip dependencies in conda environment"""
    print(f"📦 Installing pip dependencies in {env_name}...")

    # Check for requirements file
    req_files = [
        "requirements-conda.txt",
        "requirements.txt"
    ]

    req_file = None
    for rf in req_files:
        if Path(rf).exists():
            req_file = rf
            break

    if req_file:
        print(f"📋 Installing from {req_file}...")
        if os.name == 'nt':  # Windows
            pip_cmd = f"conda run -n {env_name} pip install -r {req_file}"
        else:  # Unix-like
            pip_cmd = f"conda run -n {env_name} pip install -r {req_file}"

        run_command(pip_cmd, shell=True)
    else:
        print("⚠️ No requirements file found, installing essential packages...")
        essential_packages = [
            "optuna>=3.4.0",
            "higher>=0.2.1",
            "open3d>=0.18.0",
            "laspy>=2.5.0",
            "sentence-transformers>=2.2.2"
        ]

        for package in essential_packages:
            run_command(f"conda run -n {env_name} pip install {package}", shell=True)

    print("✅ Pip dependencies installed")


def verify_installation(env_name):
    """Verify the installation"""
    print(f"🔍 Verifying installation in {env_name}...")

    verification_script = 