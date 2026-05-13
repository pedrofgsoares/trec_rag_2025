# Operator Setup — Phase 1 Prerequisites

These steps are **operator-only** (require sudo, host-Windows access, or shell restart) and must be completed before the pipeline can run end-to-end. Tasks reference the IDs in `openspec/changes/add-baseline-pipeline/tasks.md`.

Status (updated 2026-05-13):

| Task | Status |
|---|---|
| 1.1 WSL2 RAM `.wslconfig` | **DONE** |
| 1.2 Verify post-restart `free -h` | **DONE** |
| 1.3 OpenJDK 21 | **DONE** |
| 1.4 Python 3.11 + `uv venv` | **DONE** |
| 1.5 `nvidia-smi` reports Quadro T1000 4 GB | **DONE** — driver 573.44, CUDA 12.8 visible |

---

## 1.1 Bump WSL2 memory (host Windows)

On the **Windows host**, edit `C:\Users\<your-user>\.wslconfig` (create it if missing) to contain:

```ini
[wsl2]
memory=12GB
swap=8GB
```

Then from a Windows PowerShell:

```powershell
wsl --shutdown
```

Re-open the WSL Ubuntu terminal afterwards. The pipeline preflight will fail-fast if `psutil.virtual_memory().total < 11 GiB` (see `design.md` risks section).

## 1.2 Verify RAM inside WSL

```bash
free -h
# expect: "Mem:           12Gi …" or ≥11 GiB total
```

## 1.3 Install OpenJDK 21

Required by Pyserini's embedded Lucene.

```bash
sudo apt update
sudo apt install -y openjdk-21-jdk-headless
java -version  # expect: openjdk version "21.x"
```

Then export `JAVA_HOME` (add to `~/.bashrc`):

```bash
echo 'export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))' >> ~/.bashrc
source ~/.bashrc
```

## 1.4 Python 3.11 + uv venv

`uv` manages the Python toolchain directly — no `pyenv` required. It downloads
a standalone CPython 3.11 build into `~/.local/share/uv/python/` and the venv
picks it up automatically.

```bash
cd /home/up746872/projects/trec_rag_2025
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
python --version          # expect: Python 3.11.x
uv pip install -e .       # installs the project + pinned deps from pyproject.toml
```

> **Alternative (pyenv, optional):** the original spec referenced `pyenv`.
> If you prefer it for shell-wide version management, the equivalent is:
> ```bash
> curl https://pyenv.run | bash
> # add the printed lines to ~/.bashrc, then:
> exec "$SHELL"
> pyenv install 3.11.9 && pyenv local 3.11.9
> uv venv --python 3.11.9 && source .venv/bin/activate
> uv pip install -e .
> ```
> Functionally equivalent. Pick one.

## 2.3 Install scispaCy `en_core_sci_sm`

Done **after** §1.4 (needs the venv active):

```bash
source .venv/bin/activate
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
python -c "import spacy; nlp = spacy.load('en_core_sci_sm'); print('OK', nlp.meta['version'])"
```

## 2.5 Init git repo

```bash
cd /home/up746872/projects/trec_rag_2025
git init
git add -A
git commit -m "scaffold: Phase 1 baseline pipeline"
```

---

## Long-running operator gates (bucket C)

| Task | What you run | Expected wall-clock |
|---|---|---|
| 3.1 + 3.2 | `bash scripts/download_pubmed.sh` | 30–60 min on a fast link (~30 GB) |
| 3.5 | `python -m trec_biogen.ingest.parse_pubmed --all` | 1–2 h CPU |
| 4.2 | `bash scripts/build_indexes.sh` (overnight) | ≈ 12 h |
| 6.5 | `bash scripts/baseline_check.sh` | hours; **MUST pass** before any optimisation |
| 10.4 | `python -m trec_biogen.pipeline.run_task_a run=phase1_baseline` | 6–10 h full pipeline |

Each script is idempotent and logs to `runs/<id>/`. Re-running is safe.
