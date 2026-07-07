# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Vendored, version-guarded replacement for the site-packages patch of
pypowsybl's grid2op backend ``update_integer_value`` (review finding M5).

Background
----------
``pypowsybl2grid.PyPowSyBlBackend`` delegates topology writes to an internal
``pypowsybl.grid2op.Backend`` (``pypowsybl/grid2op/impl/backend.py``); it is
that class — NOT ``PyPowSyBlBackend`` itself — whose ``update_integer_value``
forwards the grid2op bus array straight to the native
``_pypowsybl.update_grid2op_integer_value``. grid2op encodes the
"disconnected / unset" bus sentinel as ``0`` while pypowsybl expects ``-1``, so
the raw forward corrupts topology.

The historical fix (``scripts/patch_pypowsybl2grid_file.py``) edited the
installed third-party file in place — load-bearing, unguarded against upstream
drift, and mutating a shared venv. This module applies the *same* one-line
correction (``value[value == 0] = -1`` before the native call) at import time
instead: no site-packages file is touched, the wrap is idempotent, and a
best-effort version guard warns if the upstream body ever changes.

The package's ``__init__.py`` already patches a third-party class at import
time (``grid2op.Backend.get_shunt_setpoint``); this mirrors that precedent.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Sentinel attribute stamped on the patched class so we never double-wrap.
_PATCH_FLAG = "_eo4g_integer_value_patched"

# The upstream body we assume (normalized): a single native passthrough call.
_ASSUMED_CALL = "update_grid2op_integer_value"


def _resolve_pp_grid2op_backend_cls():
    """Return the ``pypowsybl.grid2op.Backend`` class, or ``None`` if absent.

    This is exactly the class ``pypowsybl2grid`` instantiates internally as
    ``self._grid`` and dispatches ``update_integer_value`` on.
    """
    try:
        import pypowsybl as pp
        return pp.grid2op.Backend
    except Exception as exc:  # ImportError, or a partial/namespace install
        _logger.debug(
            "pypowsybl grid2op backend unavailable, skip integer-value patch: %s", exc
        )
        return None


def _upstream_body_matches_assumption(cls) -> bool:
    """Best-effort version guard.

    ``True`` iff the upstream ``update_integer_value`` is still the single native
    passthrough we assume (and is not already applying the 0->-1 fix itself).
    Returns ``False`` when the source can't be read (namespace/zip installs) so
    the caller warns rather than trusting an unverified body.
    """
    import inspect
    try:
        src = inspect.getsource(cls.update_integer_value)
    except (OSError, TypeError) as exc:
        _logger.debug("Cannot read upstream update_integer_value source: %s", exc)
        return False
    normalized = " ".join(src.split())
    already_fixed = "value[value == 0]" in normalized or "value[value==0]" in normalized
    single_call = normalized.count(_ASSUMED_CALL) == 1
    return single_call and not already_fixed


def apply_pypowsybl_integer_value_patch() -> bool:
    """Idempotently wrap ``pypowsybl.grid2op.Backend.update_integer_value``.

    Rewrites the grid2op ``0`` bus-sentinel to ``-1`` before the native call.
    Returns ``True`` if the patch is in force afterwards, ``False`` if pypowsybl
    is unavailable. Safe to call any number of times, and must run before any
    backend topology write (i.e. before backend construction).
    """
    cls = _resolve_pp_grid2op_backend_cls()
    if cls is None:
        return False
    if getattr(cls, _PATCH_FLAG, False):
        return True  # already patched by us

    if not _upstream_body_matches_assumption(cls):
        try:
            import pypowsybl as pp
            _ver = getattr(pp, "__version__", "?")
        except Exception:
            _ver = "?"
        _logger.warning(
            "pypowsybl %s update_integer_value body differs from the assumed single "
            "passthrough (or already applies the 0->-1 fix). Applying the conversion "
            "anyway (idempotent), but review patched_backend.py "
            "against the installed pypowsybl before trusting results.",
            _ver,
        )

    _orig = cls.update_integer_value

    def _patched(self, value_type, value, changed):
        # grid2op 0-sentinel (disconnected / unset bus) -> pypowsybl -1.
        # Idempotent: after this no zeros remain, so re-applying is a no-op.
        value[value == 0] = -1
        return _orig(self, value_type, value, changed)

    _patched.__name__ = "update_integer_value"
    _patched.__qualname__ = f"{cls.__qualname__}.update_integer_value"
    cls.update_integer_value = _patched
    setattr(cls, _PATCH_FLAG, True)
    _logger.debug("Applied pypowsybl update_integer_value 0->-1 patch to %s", cls)
    return True


def _load_pypowsybl2grid_backend_cls():
    """Lazily resolve ``pypowsybl2grid.PyPowSyBlBackend`` (or ``None``)."""
    try:
        from pypowsybl2grid import PyPowSyBlBackend
        return PyPowSyBlBackend
    except Exception as exc:  # ImportError or partial install
        _logger.debug("pypowsybl2grid unavailable: %s", exc)
        return None


def make_patched_pypowsybl_backend(*args, **kwargs):
    """Construct a ``pypowsybl2grid.PyPowSyBlBackend`` with the integer-value
    patch guaranteed in force first.

    This is the factory entry point (the review's "importable / testable"
    surface). The buggy method lives on the internal ``pypowsybl.grid2op.Backend``
    delegate, so there is nothing useful to override on ``PyPowSyBlBackend``
    itself — instead we guarantee the process-wide class patch before building
    the backend. Raises ``ImportError`` if pypowsybl2grid is not installed.

    .. deprecated:: 0.3.0
        ``pypowsybl2grid`` is deprecated and no longer a dependency (its
        ``numpy==1.26.4`` pin conflicts with ``numpy>=2.0.0``). Install it
        manually into a ``numpy<2`` environment if you still need this factory.
        The generic ``apply_pypowsybl_integer_value_patch()`` remains active.
    """
    backend_cls = _load_pypowsybl2grid_backend_cls()
    if backend_cls is None:
        raise ImportError(
            "pypowsybl2grid is deprecated and no longer a dependency; install it "
            "manually in a numpy<2 environment to build a PyPowSyBlBackend."
        )
    apply_pypowsybl_integer_value_patch()
    return backend_cls(*args, **kwargs)
