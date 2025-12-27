# CLAUDE.md

## Current Mission

**Read `AUTONOMOUS_SESSION.md` and execute the instructions.**

Master instructions are in docs/FLYWHEEL_MASTER_PLAN.md, only read as needed.

That file contains everything you need: tasks, code templates, execution order, and completion criteria.

## Project Context

**SegOid** — PyTorch U-Net pipeline for spheroid segmentation.

**POC (Phases 0-5) is complete.** Do not modify existing code in `src/` unless the autonomous session instructions tell you to.

## Quick Reference

```bash
# Environment
source .venv/bin/activate
pip install -e .

# Testing (run frequently)
pytest

# Commit pattern
git add -A && git commit -m "feat: description"
```

## Tech Stack

Python 3.11+, PyTorch, segmentation-models-pytorch, albumentations, tifffile, imagecodecs, scikit-image, pandas, PyYAML

## Code Style

- Type hints on public functions
- Docstrings for modules and classes  
- Match existing patterns in `src/`
