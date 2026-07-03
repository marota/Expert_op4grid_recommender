# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Domain exceptions for the recommender pipeline."""

from __future__ import annotations


class LoadFlowDivergedError(RuntimeError):
    """Raised when the power flow fails to converge — including the DC fallback —
    so the overflow graph cannot be built.

    Subclasses :class:`RuntimeError` so the CLI ``main()`` handler (which maps
    ``RuntimeError`` to a non-zero exit) still catches it, while library, UI and
    notebook callers can catch it specifically and recover — instead of the
    whole host process being terminated by ``sys.exit()``.
    """
