# Installation and public data

Use a dedicated Python 3.12 environment. Full training needs CUDA GPUs; the allocation example and CPU tests need Python and pytest only.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/setup_external.py
pip install -e external/verl --no-deps
pip install -e external/plancraft
pip install -r requirements-environments.txt
source env.sh
mkdir -p cache data
cc -shared -fPIC -O2 scripts/nvml_selected_devices.c -o cache/nvml_selected_devices.so -ldl
```

`setup_external.py` fetches pinned upstream commits and applies only the SFT-related verl changes. It refuses to silently move an existing checkout to a different revision. Install any additional dependencies requested by the upstream verl package for your hardware. `requirements.txt` records the central training versions; it is not a solver lock for every environment. Use a separate ALFWorld interpreter via `--environment-python` if its dependency constraints conflict with the training interpreter.

## Models and common SFT

Download `meta-llama/Llama-3.2-1B-Instruct` or `Qwen/Qwen3-4B-Instruct-2507` under their model terms. Set `DCHL_BASE_MODEL` to a local checkpoint directory. The data-preparation commands below tokenize with that model; keep model-specific prepared data and fixed templates separate when switching backbones.

## ALFWorld

```bash
python tools/fetch_demonstrations.py alfworld
hf download zhangdw/alfworld --repo-type dataset --local-dir external/alfworld_text_data
# Extract the dataset's raw/alfworld.tar.gz using its documented setup.
# Set ALFWORLD_DATA to the resulting directory containing json_2.1.1/.
python scripts/prepare_alfworld_eto_sft800.py
python scripts/prepare_alfworld_action_contract.py
python scripts/prepare_alfworld_dialogue_hsl.py
```

The demonstration selection fixes 800 public ETO trajectories. The action-contract builder creates whole-demonstration SFT inputs and per-action online records. The collection panel uses the native TRAIN task manifest; evaluation uses the separate seen/unseen manifests. No evaluation trajectory enters the reference bank.

## PlanCraft

The pinned upstream checkout supplies the official feasible-task splits and oracle training traces. Check that `external/plancraft/oracle_trajectories/train/oa/` contains the upstream demonstration files before preparation. Follow upstream asset download instructions if your checkout does not include them.

```bash
python scripts/build_plancraft_text_data.py
python scripts/prepare_plancraft_hsl.py
```

This replays the oracle actions in the native environment and freezes recipes. For online training, export the **common SFT checkpoint's** chat template to `data/plancraft_text_v1/fixed_chat_template.jinja`; the Llama structural-feedback recipe used the fixed date `02 Oct 2026`. `tools/train.py --chat-template PATH` can supply a different recorded template explicitly. Keep the template identical across the compared methods.

## BabyAI

```bash
python tools/fetch_demonstrations.py babyai
python scripts/prepare_babyai.py
```

The builder reconstructs 810 public training demonstrations and 90 evaluation tasks in the native environment, checking terminal success and goal ordering. It writes the fixed template, online records and TRAIN-only replay references. The historical Llama preparation fixed the template date to `01 Nov 2026`.

The public download URLs use upstream dataset branches. `fetch_demonstrations.py` prints SHA-256 digests; retain them with your experiment. A changed upstream dataset needs an explicit new data version, not silent replacement of a comparison.

## Optional GitHub CI

`docs/ci-workflow.example.yml` is the tested CPU-check workflow template. It is not active: the publication credential has repository access but no `workflow` scope. An authorized maintainer can install it under `.github/workflows/tests.yml`. The local CPU tests remain available without GitHub Actions.
