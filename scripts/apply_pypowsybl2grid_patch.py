#!/usr/bin/env python3
"""
DEPRECATED: This runtime patching approach does not work reliably.

Use scripts/patch_pypowsybl2grid_file.py instead to patch the installed
package file directly before running tests.

Usage:
    python scripts/patch_pypowsybl2grid_file.py
"""

import sys

print("=" * 60)
print("DEPRECATED: Runtime patching does not work")
print("=" * 60)
print()
print("The pypowsybl2grid backend is imported before this patch can")
print("be applied. Use the file-based patcher instead:")
print()
print("    python scripts/patch_pypowsybl2grid_file.py")
print()
print("This will modify the installed pypowsybl2grid package directly.")
print("=" * 60)

sys.exit(1)
