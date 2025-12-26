#!/bin/bash
# run_autonomous_session.sh
# Execute this before leaving to tag the repo and start Claude Code

set -e

echo "=========================================="
echo "SegOid Autonomous Session Setup"
echo "=========================================="

# Navigate to project (adjust path as needed)
# cd /path/to/segoid

# Activate environment
source .venv/bin/activate

# Ensure clean git state
echo "Checking git status..."
if [[ -n $(git status --porcelain) ]]; then
    echo "WARNING: Uncommitted changes detected. Committing..."
    git add -A
    git commit -m "chore: pre-autonomous-session snapshot"
fi

# Tag the current state
TAG_NAME="pre-autonomous-phase6-$(date +%Y%m%d_%H%M%S)"
echo "Creating tag: $TAG_NAME"
git tag -a "$TAG_NAME" -m "Snapshot before autonomous Phase 6 implementation"

echo "Tag created. To restore this state later:"
echo "  git checkout $TAG_NAME"
echo ""

# Run tests to confirm starting state is good
echo "Running tests to confirm clean starting state..."
pytest --tb=short
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failing before autonomous session. Fix first."
    exit 1
fi

echo ""
echo "=========================================="
echo "Ready for autonomous session!"
echo "=========================================="
echo ""
echo "Instructions for Claude Code:"
echo "1. Open Claude Code in this project directory"
echo "2. Tell it: 'Read AUTONOMOUS_SESSION.md and execute the instructions'"
echo "3. Walk away"
echo ""
echo "To monitor progress remotely, check git commits:"
echo "  git log --oneline -10"
echo ""
echo "To rollback if needed:"
echo "  git checkout $TAG_NAME"
echo "  git checkout -b recovery-branch"
echo ""
