"""
Setup configuration for Real-Time Bidirectional Sign Language Translation System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sign-language-translation",
    version="1.0.0",
    author="Sign Language Translation Team",
    description="Real-time bidirectional sign language translation system with offline speech recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/real-time-sign-language-translation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "customtkinter>=5.2.0",
        "vosk>=0.3.32",
        "pyttsx3>=2.90",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
        "requests>=2.31.0",
        "pyaudio>=0.2.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "pylint>=2.17.0",
            "mypy>=1.5.0",
            "tqdm>=4.66.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
)
