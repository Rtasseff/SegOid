# Configuration Files

This directory contains YAML configuration files for each phase of the pipeline.

Configuration files will be created during implementation of each phase:

- `dataset.yaml` - Patch sampling and augmentation (Phase 2)
- `sanity_check.yaml` - Pipeline validation parameters (Phase 1.5)
- `train.yaml` - Model architecture and training hyperparameters (Phase 3)
- `predict.yaml` - Tiled inference parameters (Phase 4)
- `quantify.yaml` - Object analysis and matching parameters (Phase 5)

See `docs/SDD.md` for detailed specifications of each configuration file.
