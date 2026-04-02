# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
ExpertOp4Grid Recommender - Expert system for power grid contingency analysis.

Analyzes N-1 contingencies in Grid2Op/pypowsybl environments, builds overflow
graphs, applies expert rules to filter potential actions, and identifies
corrective measures to alleviate line overloads.
"""

__version__ = "0.1.9"

# --- GLOBAL MONKEY PATCH for grid2op 1.12+ compatibility ---
# PyPowSyBlBackend 0.3.0 and other backends might not initialize self._sh_vnkv,
# which is required by grid2op 1.12+ for grids with shunts.
try:
    import sys
    import numpy as np
    import grid2op.Backend.backend as g2op_bk
    _orig_get_shunt_setpoint = g2op_bk.Backend.get_shunt_setpoint
    def _patched_get_shunt_setpoint(self):
        if (not hasattr(self, "_sh_vnkv") or self._sh_vnkv is None) and getattr(self, "n_shunt", 0) > 0:
            print(f"DEBUG: Initializing _sh_vnkv for {type(self)}", file=sys.stderr)
            self._sh_vnkv = np.ones(self.n_shunt, dtype=np.float32) * 225.0
        return _orig_get_shunt_setpoint(self)
    g2op_bk.Backend.get_shunt_setpoint = _patched_get_shunt_setpoint
    print("DEBUG: grid2op.Backend.Backend.get_shunt_setpoint patched", file=sys.stderr)
except (ImportError, Exception) as e:
    try:
        import sys
        print(f"DEBUG: Failed to patch grid2op: {e}", file=sys.stderr)
    except:
        pass
# -----------------------------------------------------------
