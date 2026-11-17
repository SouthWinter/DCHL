# Source this file from your chosen Python environment.
export DCHL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DCHL_VERL_ROOT="${DCHL_VERL_ROOT:-$DCHL_ROOT/external/verl}"
export DCHL_PYTHON="${DCHL_PYTHON:-$(command -v python)}"
export PYTHONPATH="$DCHL_ROOT/scripts:$DCHL_VERL_ROOT:$DCHL_ROOT/external/plancraft${PYTHONPATH:+:$PYTHONPATH}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$DCHL_ROOT/external/alfworld_text_data/.alfworld_data/alfworld}"
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
