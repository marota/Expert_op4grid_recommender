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

__version__ = "0.1.1"
