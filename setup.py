from setuptools import setup, find_packages

setup(
    name="aquaintel",
    version="1.0.0",
    description="Water Scarcity Prediction & Intervention System — Ganges-Brahmaputra Basin",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "networkx>=3.1",
        "loguru>=0.7.2",
        "pyyaml>=6.0.1",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "geopandas>=0.14.0",
        "shap>=0.43.0",
        "dowhy>=0.11.0",
        "mapie>=0.7.0",
        "pettingzoo>=1.24.0",
        "stable-baselines3>=2.2.0",
        "gymnasium>=0.29.0",
        "einops>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "aquaintel=main:main",
        ]
    },
)
