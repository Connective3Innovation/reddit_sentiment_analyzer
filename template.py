# scaffold.py
import os
import logging
from pathlib import Path

# ──────────────────────────────
# Logging config
# ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s: %(message)s"
)

# Project-specific root package name
project_name = "reddit_sentiment"

# ──────────────────────────────
# Folder & file blueprint
# ──────────────────────────────
files = [
    # CI / CD
    ".github/workflows/ci.yml",

    # ╔══════════════════════╗
    # ║ Core application src ║
    # ╚══════════════════════╝
    f"src/{project_name}/__init__.py",

    # ‣ Authentication & API client
    f"src/{project_name}/auth/__init__.py",
    f"src/{project_name}/auth/reddit_auth.py",

    # ‣ Configuration management
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/settings.py",

    # ‣ Low-level Reddit API helpers
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/reddit_client.py",

    # ‣ Data collection & preprocessing
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/collector.py",
    f"src/{project_name}/data/preprocess.py",

    # ‣ Sentiment engines
    f"src/{project_name}/sentiment/__init__.py",
    f"src/{project_name}/sentiment/vader_engine.py",
    f"src/{project_name}/sentiment/hf_engine.py",

    # ‣ ETL / orchestration
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/etl.py",

    # Command-line entrypoints
    f"src/{project_name}/cli.py",
    f"src/{project_name}/main.py",

    # ╔════════════╗
    # ║ App assets ║
    # ╚════════════╝
    "config/.gitkeep",           # external YAML / TOML files live here
    ".env",                      # environment variables (never commit secrets!)

    # ╔═══════════════╗
    # ║ Documentation ║
    # ╚═══════════════╝
    "README.md",

    # ╔══════════════╗
    # ║ Packaging    ║
    # ╚══════════════╝
    "pyproject.toml",            # modern build / dependency metadata
    "setup.cfg",                 # optional for flake8 / isort
    "requirements.txt",
    ".gitignore",

    # ╔══════════╗
    # ║ Testing  ║
    # ╚══════════╝
    "tests/__init__.py",
    "tests/test_api.py",
    "tests/test_sentiment.py",
    "tests/test_pipeline.py",
]

# ──────────────────────────────
# Create directories & files
# ──────────────────────────────
for path in files:
    path = Path(path)
    dir_name, file_name = os.path.split(path)

    # Make parent directories if needed
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        logging.info(f"Directory created: {dir_name}")

    # Touch empty placeholder files
    if not path.exists() or path.stat().st_size == 0:
        path.touch()
        logging.info(f"Empty file created: {path}")

logging.info("🎉  Project structure created successfully!")
