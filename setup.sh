#!/usr/bin/env bash
# One-shot setup for a fresh CPU VM (GCP / Debian or Ubuntu x86).
#
# Before running:
#   1. push this repo to GitHub and set REPO_URL below
#   2. upload the ontology to the VM, e.g.:
#        gcloud compute scp thesisont_updated-2.owl VM_NAME:~/   (or scp ...)
#   3. bash setup.sh
#
# Then start training (parallel, survives SSH disconnect):
#   tmux new -s train
#   source ~/Thesis/.venv/bin/activate
#   ONTOLOGY_PATH=~/thesisont_updated-2.owl N_ENVS=$(nproc) python train_scripts/train_ppo.py

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/CHANGE_ME/Thesis.git}"
REPO_DIR="${REPO_DIR:-$HOME/Thesis}"
ONT_PATH="${ONT_PATH:-$HOME/thesisont_updated-2.owl}"

echo "==> system packages (Java for HermiT, git, python, tmux)"
sudo apt-get update -y
sudo apt-get install -y default-jdk git python3-venv python3-pip build-essential tmux

export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(which java)")")")"
echo "JAVA_HOME=$JAVA_HOME"
java -version

echo "==> clone repo"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "repo already present at $REPO_DIR, pulling latest"
    git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"

echo "==> python venv + deps"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU torch first (avoids pulling the multi-GB CUDA build on a CPU VM)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
  || pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "==> checks"
if [ ! -f "$ONT_PATH" ]; then
    echo "WARNING: ontology not found at $ONT_PATH"
    echo "  upload it, then export ONTOLOGY_PATH to its path before training."
else
    echo "ontology found: $ONT_PATH"
fi
python - <<'PY'
import torch, stable_baselines3, gymnasium, owlapy, jpype, multigrid
print("torch", torch.__version__, "| sb3", stable_baselines3.__version__,
      "| cores", __import__("os").cpu_count())
PY

cat <<EOF

==> setup done. Start training with:

  cd $REPO_DIR
  source .venv/bin/activate
  tmux new -s train
  ONTOLOGY_PATH=$ONT_PATH N_ENVS=\$(nproc) python train_scripts/train_ppo.py

  # detach from tmux: Ctrl-b then d   |   reattach: tmux attach -t train

Smoke-test first (confirms HermiT runs in the loop on this VM):
  ONTOLOGY_PATH=$ONT_PATH N_ENVS=2 python train_scripts/smoke_reasoner.py
EOF
