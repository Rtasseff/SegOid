# AUTONOMOUS_SESSION.md

**Project:** SegOid  
**Created:** 2025-12-27  
**Purpose:** Instructions for Claude Code to work autonomously on Phase 6

---

## Mission

Implement Phase 6 (Cross-Validation Infrastructure) completely while the user is away. Write code, write tests, run tests, fix bugs, iterate until everything works.

---

## Ground Rules

1. **Read first, code second.** Before writing any code, read:
   - `CURRENT_TASK.md` — Phase 6 tasks and completion criteria
   - `FLYWHEEL_MASTER_PLAN.md` — Context for what you're building
   - `CLAUDE.md` — Project conventions and commands

2. **Test everything.** After implementing each function, write a unit test. Run `pytest` frequently. Do not move on if tests fail.

3. **Small commits, clear messages.** Commit after each logical unit of work:
   - `git add -A && git commit -m "feat: implement generate_loocv_splits"`
   - `git add -A && git commit -m "test: add unit tests for LOOCV split generation"`
   - `git add -A && git commit -m "fix: handle edge case in fold config generation"`

4. **Follow existing patterns.** Look at how Phase 3-5 code is structured:
   - `src/training/train.py` — training patterns
   - `src/inference/predict.py` — inference patterns
   - `src/cli.py` — CLI patterns
   - Match the style, imports, logging, error handling.

5. **Update CURRENT_TASK.md** as you complete tasks. Check off boxes.

6. **If stuck, document and move on.** If something is blocked or unclear:
   - Add a note to CURRENT_TASK.md Notes section
   - Move to the next task
   - The user will resolve blockers on return

7. **Do not skip folds.** Run all 6 folds of leave-one-out CV. Each fold will train until early stopping triggers (plateau). This may take 4-6 hours total—that's fine, let it run.

---

## Task Execution Order

### Step 1: Setup
```bash
cd /path/to/segoid  # adjust to actual project path
source .venv/bin/activate
git status  # confirm clean state
```

### Step 2: Create cross-validation split generation
Create `src/data/cross_validation.py`:
- `generate_loocv_splits(manifest_path, output_dir, seed)` → List of (train_csv, val_csv) paths
- `generate_kfold_splits(manifest_path, output_dir, n_folds, seed)` → same
- Save `cv_meta.yaml` with fold metadata

Test it:
```bash
pytest tests/test_cross_validation.py -v
```

Commit:
```bash
git add -A && git commit -m "feat: implement CV split generation (LOOCV and k-fold)"
```

### Step 3: Create CV configuration
Create `configs/cv_config.yaml` with the schema from CURRENT_TASK.md.

Commit:
```bash
git add -A && git commit -m "feat: add CV config template"
```

### Step 4: Create CV orchestrator
Create `src/training/cross_validation.py`:
- `load_cv_config(path)` — load and validate CV config
- `create_fold_config(base_config, train_csv, val_csv, output_dir)` — generate per-fold train config
- `run_cross_validation(cv_config_path, folds=None)` — main orchestration function
- `aggregate_results(fold_results)` — compute mean, std, etc.

This should import and call `train_model()` from `src/training/train.py`.

Test it:
```bash
pytest tests/test_cross_validation.py -v
```

Commit:
```bash
git add -A && git commit -m "feat: implement CV orchestrator"
```

### Step 5: Add CLI command
Update `src/cli.py`:
- Add `run_cv` command with `--config`, `--folds`, `--resume` parameters

Test it manually:
```bash
python -m src.cli run_cv --help
```

Commit:
```bash
git add -A && git commit -m "feat: add run_cv CLI command"
```

### Step 6: Run full 6-fold CV
Run the complete cross-validation:
```bash
python -m src.cli run_cv --config configs/cv_config.yaml
```

This will:
- Create `runs/cv_<timestamp>/`
- Train all 6 folds (each until early stopping plateau)
- Generate `fold_metrics.csv` and `summary.yaml`
- Total time: ~4-6 hours

Let it run to completion. Do not interrupt.

Commit after completion:
```bash
git add -A && git commit -m "results: complete 6-fold LOOCV experiment"
```

### Step 7: Generate summary report
After CV completes, create a markdown report summarizing results.

Create `runs/cv_<experiment_id>/results/REPORT.md`:

```python
# Add this function to src/training/cross_validation.py or create a separate script

def generate_cv_report(cv_dir: Path) -> None:
    """Generate a markdown summary report for CV experiment."""
    import pandas as pd
    import yaml
    from datetime import datetime
    
    results_dir = cv_dir / "results"
    fold_metrics = pd.read_csv(results_dir / "fold_metrics.csv")
    
    with open(results_dir / "summary.yaml") as f:
        summary = yaml.safe_load(f)
    
    report = f"""# Cross-Validation Report

**Experiment:** {cv_dir.name}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Val Dice | {summary['val_dice']['mean']:.4f} | {summary['val_dice']['std']:.4f} | {summary['val_dice']['min']:.4f} | {summary['val_dice']['max']:.4f} |

## Per-Fold Results

| Fold | Val Image | Best Val Dice | Best Epoch | Training Time |
|------|-----------|---------------|------------|---------------|
"""
    
    for _, row in fold_metrics.iterrows():
        report += f"| {int(row['fold'])} | {row['val_image']} | {row['best_val_dice']:.4f} | {int(row['best_epoch'])} | {row['training_time_min']:.1f} min |\n"
    
    report += f"""
## Analysis

- **Best performing fold:** {summary['best_fold']} (Dice: {fold_metrics.loc[fold_metrics['fold']==summary['best_fold'], 'best_val_dice'].values[0]:.4f})
- **Worst performing fold:** {summary['worst_fold']} (Dice: {fold_metrics.loc[fold_metrics['fold']==summary['worst_fold'], 'best_val_dice'].values[0]:.4f})
- **Total training time:** {summary.get('total_training_time_min', 0):.1f} minutes

## Conclusion

Cross-validation complete with {summary['n_folds']} folds.  
Expected generalization performance: **{summary['val_dice']['mean']:.4f} ± {summary['val_dice']['std']:.4f}** Dice score.
"""
    
    with open(results_dir / "REPORT.md", "w") as f:
        f.write(report)
    
    print(f"Report saved to {results_dir / 'REPORT.md'}")
```

Call this function at the end of `run_cross_validation()` or run it separately after CV completes.

Commit:
```bash
git add -A && git commit -m "feat: add CV summary report generation"
```

### Step 8: Update documentation
- Add implementation notes to the Notes section of any task files
- Update `CLAUDE.md` if any new commands or conventions were added

Commit:
```bash
git add -A && git commit -m "docs: update documentation with CV implementation notes"
```

### Step 9: Final test suite
```bash
pytest --cov=src tests/ -v
```

All tests should pass. If any fail, fix them before stopping.

Final commit:
```bash
git add -A && git commit -m "chore: Phase 6 complete - 6-fold LOOCV with report"
```

---

## Code Templates

### src/data/cross_validation.py (starter)

```python
"""Cross-validation split generation utilities."""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import yaml


def generate_loocv_splits(
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42
) -> List[Tuple[Path, Path]]:
    """
    Generate leave-one-out cross-validation splits.
    
    Args:
        manifest_path: Path to source manifest CSV (all images)
        output_dir: Directory to write fold manifests
        seed: Random seed (not used for LOOCV, but kept for API consistency)
    
    Returns:
        List of (train_csv_path, val_csv_path) tuples, one per fold
    """
    df = pd.read_csv(manifest_path)
    n_images = len(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fold_paths = []
    fold_metadata = {
        "strategy": "leave_one_out",
        "n_folds": n_images,
        "seed": seed,
        "source_manifest": str(manifest_path),
        "folds": []
    }
    
    for fold_idx in range(n_images):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        # Leave one out
        val_df = df.iloc[[fold_idx]].reset_index(drop=True)
        train_df = df.drop(index=fold_idx).reset_index(drop=True)
        
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        fold_paths.append((train_path, val_path))
        fold_metadata["folds"].append({
            "fold": fold_idx,
            "val_image": df.iloc[fold_idx]["basename"],
            "n_train": len(train_df),
            "n_val": len(val_df)
        })
    
    # Save metadata
    with open(output_dir / "cv_meta.yaml", "w") as f:
        yaml.dump(fold_metadata, f, default_flow_style=False)
    
    return fold_paths


def generate_kfold_splits(
    manifest_path: Path,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[Path, Path]]:
    """
    Generate k-fold cross-validation splits.
    
    Args:
        manifest_path: Path to source manifest CSV
        output_dir: Directory to write fold manifests
        n_folds: Number of folds
        seed: Random seed for shuffling
    
    Returns:
        List of (train_csv_path, val_csv_path) tuples, one per fold
    """
    # TODO: Implement k-fold splitting
    # Use sklearn.model_selection.KFold or implement manually
    raise NotImplementedError("k-fold splitting not yet implemented")
```

### src/training/cross_validation.py (starter)

```python
"""Cross-validation orchestration."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml
import pandas as pd

from src.data.cross_validation import generate_loocv_splits, generate_kfold_splits
from src.training.train import train_model, load_config, save_config


def load_cv_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate CV configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ["cv", "training", "output"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config


def create_fold_config(
    base_training_config: Dict[str, Any],
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Create a fold-specific training configuration.
    
    Merges base training config with fold-specific paths.
    """
    fold_config = {
        "model": base_training_config.get("model", {
            "architecture": "unet",
            "encoder": "resnet18",
            "pretrained": True
        }),
        "training": {
            **base_training_config,
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
        },
        "output": {
            "run_dir": str(output_dir),
        }
    }
    
    # Remove nested keys that shouldn't be in training
    for key in ["model", "output"]:
        fold_config["training"].pop(key, None)
    
    return fold_config


def run_cross_validation(
    cv_config_path: Path,
    folds: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Run full cross-validation experiment.
    
    Args:
        cv_config_path: Path to CV configuration YAML
        folds: Optional list of specific fold indices to run (for debugging)
    
    Returns:
        Dictionary with aggregated results
    """
    config = load_cv_config(cv_config_path)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = Path(config["output"]["cv_dir"])
    if not cv_dir.is_absolute():
        cv_dir = Path("runs") / f"cv_{timestamp}"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot
    save_config(config, cv_dir / "cv_config.yaml")
    
    # Generate splits
    source_manifest = Path(config["cv"]["source_manifest"])
    strategy = config["cv"].get("strategy", "leave_one_out")
    seed = config["cv"].get("seed", 42)
    
    folds_dir = cv_dir / "folds"
    
    if strategy == "leave_one_out":
        fold_paths = generate_loocv_splits(source_manifest, folds_dir, seed)
    elif strategy == "k_fold":
        n_folds = config["cv"].get("n_folds", 5)
        fold_paths = generate_kfold_splits(source_manifest, folds_dir, n_folds, seed)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")
    
    # Filter to specific folds if requested
    if folds is not None:
        fold_paths = [(fold_paths[i][0], fold_paths[i][1]) for i in folds]
        fold_indices = folds
    else:
        fold_indices = list(range(len(fold_paths)))
    
    # Run training for each fold
    fold_results = []
    
    for fold_idx, (train_csv, val_csv) in zip(fold_indices, fold_paths):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(fold_paths)} (index {fold_idx})")
        print(f"Training on: {train_csv}")
        print(f"Validating on: {val_csv}")
        print(f"{'='*60}\n")
        
        fold_output_dir = folds_dir / f"fold_{fold_idx}"
        
        # Create fold-specific config
        fold_config = create_fold_config(
            base_training_config=config["training"],
            train_manifest=train_csv,
            val_manifest=val_csv,
            output_dir=fold_output_dir
        )
        
        # Save fold config
        save_config(fold_config, fold_output_dir / "config.yaml")
        
        # Train (reuse existing train_model function)
        # NOTE: You may need to adapt this call based on how train_model() works
        results = train_model(fold_config)
        
        fold_results.append({
            "fold": fold_idx,
            "val_image": val_csv.stem,  # or extract from manifest
            "best_val_dice": results.get("best_val_dice"),
            "best_val_iou": results.get("best_val_iou"),
            "best_epoch": results.get("best_epoch"),
            "final_train_dice": results.get("final_train_dice"),
            "training_time_min": results.get("training_time_min"),
        })
    
    # Aggregate results
    results_dir = cv_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    
    summary = aggregate_results(fold_results)
    summary["experiment_dir"] = str(cv_dir)
    
    with open(results_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Folds completed: {len(fold_results)}")
    print(f"Val Dice: {summary['val_dice']['mean']:.4f} ± {summary['val_dice']['std']:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    return summary


def aggregate_results(fold_results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics across folds."""
    df = pd.DataFrame(fold_results)
    
    def compute_stats(series):
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    
    summary = {
        "n_folds": len(fold_results),
        "val_dice": compute_stats(df["best_val_dice"]),
        "best_fold": int(df.loc[df["best_val_dice"].idxmax(), "fold"]),
        "worst_fold": int(df.loc[df["best_val_dice"].idxmin(), "fold"]),
    }
    
    if "best_val_iou" in df.columns and df["best_val_iou"].notna().any():
        summary["val_iou"] = compute_stats(df["best_val_iou"])
    
    if "training_time_min" in df.columns and df["training_time_min"].notna().any():
        summary["total_training_time_min"] = float(df["training_time_min"].sum())
    
    return summary
```

---

## Checklist Before User Returns

- [ ] `src/data/cross_validation.py` exists and has tests
- [ ] `src/training/cross_validation.py` exists and has tests
- [ ] `configs/cv_config.yaml` exists
- [ ] `run_cv` CLI command works (`run_cv --help` shows options)
- [ ] All unit tests pass (`pytest`)
- [ ] Full 6-fold CV completed successfully
- [ ] `runs/cv_<id>/results/fold_metrics.csv` contains all 6 folds
- [ ] `runs/cv_<id>/results/summary.yaml` contains aggregated statistics
- [ ] `runs/cv_<id>/results/REPORT.md` contains human-readable summary
- [ ] All changes committed to git

---

## If Everything Goes Smoothly

If you finish Phase 6 with time to spare, you may begin Phase 7 (Batch Inference):

1. Read the Phase 7 section in `FLYWHEEL_MASTER_PLAN.md`
2. Create `src/inference/batch_predict.py`
3. Implement `batch_predict` CLI command
4. Do NOT start Phase 8 (Review Interface) — that requires user input on technology choice

---

## Emergency Stop Conditions

Stop and leave notes if:
- Tests fail repeatedly and you can't diagnose why
- You need to modify core training code in unexpected ways
- You're unsure about a design decision not covered in the docs
- Something seems wrong with the data or existing code

Document the issue clearly in CURRENT_TASK.md Notes section and commit what you have.