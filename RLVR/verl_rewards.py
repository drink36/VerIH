"""Load verl's reward + GRPO-advantage functions without importing verl/__init__.

verl/__init__.py pulls in ray, which we don't want in the tinker training loop.
So we load the source-of-truth modules directly:
  - verl/utils/reward_score/{ih.py, if_functions.py}  -> reward scoring
  - verl/trainer/ppo/core_algos.py                     -> GRPO advantage

Exports: ih_compute_score, check_format, check_answer, compute_grpo_outcome_advantage
"""
import importlib.util as _ilu
import sys
import types as _types
from pathlib import Path

_RS_DIR = Path(__file__).parent / "verl" / "utils" / "reward_score"

# Synthetic parent package so `from .if_functions import ...` in ih.py resolves.
_parent_spec = _ilu.spec_from_loader("_verl_rs", loader=None)
_parent = _ilu.module_from_spec(_parent_spec)
_parent.__path__ = [str(_RS_DIR)]
sys.modules["_verl_rs"] = _parent

# Load if_functions first (ih.py imports it relatively).
_iff_spec = _ilu.spec_from_file_location("_verl_rs.if_functions", _RS_DIR / "if_functions.py")
_iff = _ilu.module_from_spec(_iff_spec)
sys.modules["_verl_rs.if_functions"] = _iff
_iff_spec.loader.exec_module(_iff)

# Now ih — its relative import to .if_functions resolves via sys.modules.
_ih_spec = _ilu.spec_from_file_location("_verl_rs.ih", _RS_DIR / "ih.py")
_verl_ih = _ilu.module_from_spec(_ih_spec)
sys.modules["_verl_rs.ih"] = _verl_ih
_ih_spec.loader.exec_module(_verl_ih)

ih_compute_score = _verl_ih.compute_score
check_format = _verl_ih.check_format
check_answer = _verl_ih.check_answer

# Stub verl.utils.torch_functional (unused by compute_grpo_outcome_advantage) so
# loading core_algos.py doesn't trigger verl/__init__.py and its ray import.
sys.modules.setdefault("verl", _types.ModuleType("verl"))
sys.modules.setdefault("verl.utils", _types.ModuleType("verl.utils"))
sys.modules.setdefault("verl.utils.torch_functional", _types.ModuleType("verl.utils.torch_functional"))

_CA_PATH = Path(__file__).parent / "verl" / "trainer" / "ppo" / "core_algos.py"
_ca_spec = _ilu.spec_from_file_location("_verl_core_algos", _CA_PATH)
_verl_ca = _ilu.module_from_spec(_ca_spec)
sys.modules["_verl_core_algos"] = _verl_ca
_ca_spec.loader.exec_module(_verl_ca)

compute_grpo_outcome_advantage = _verl_ca.compute_grpo_outcome_advantage
