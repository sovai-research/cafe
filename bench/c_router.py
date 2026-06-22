"""
CausalRouter — the unified causal core. Routes to the best-performing causal
engine per data type (the leaderboard showed OnlineTRMF wins 2D, CausalFE wins
panel). This is the prototype of the library's auto-routing front-end.

Routing rule (causal, uses only structure that is known up-front, not future data):
  - panel (meta has entity_ids with >1 entity)            -> CausalFE
  - 2D temporal/tabular                                   -> OnlineTRMF
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe


def _is_panel(meta):
    if not meta or "entity_ids" not in meta:
        return False
    return len(np.unique(meta["entity_ids"])) > 1


def online_impute(X, meta):
    return _fe(X, meta) if _is_panel(meta) else _trmf(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("CausalRouter", online_impute))
