#!/usr/bin/env python3
"""
Patch script for pypowsybl2grid backend.

This script patches the PyPowSyBlBackend.update_integer_value method to fix
zero-value handling issues. It can be run:
1. Automatically via conftest.py (recommended for tests)
2. Manually before running the application
3. In CI/CD pipelines before tests

Usage:
    python scripts/apply_pypowsybl2grid_patch.py [--verify]
    
Options:
    --verify    Only verify if patch is needed/applied, don't modify anything
"""

import sys
import argparse


def check_if_patch_needed():
    """Check if the patch is needed by inspecting the source code."""
    try:
        import inspect
        from pypowsybl2grid.backend import PyPowSyBlBackend
        
        source = inspect.getsource(PyPowSyBlBackend.update_integer_value)
        
        # Check for the problematic line
        if "changed[value == 0] = False" in source:
            return True, "Original problematic code found"
        elif "value[value == 0] = -1" in source or "value[value==0] = -1" in source:
            return False, "Patch already applied in source"
        elif "Patched version" in (PyPowSyBlBackend.update_integer_value.__doc__ or ""):
            return False, "Runtime patch already applied"
        else:
            return None, "Cannot determine patch status"
            
    except ImportError as e:
        return None, f"pypowsybl2grid not installed: {e}"
    except Exception as e:
        return None, f"Error checking patch status: {e}"


def apply_runtime_patch():
    """Apply the patch at runtime by monkey-patching the method."""
    import numpy as np
    
    try:
        from pypowsybl2grid.backend import PyPowSyBlBackend
        import pypowsybl._pypowsybl as _pypowsybl
        from pypowsybl2grid.backend import Grid2opUpdateIntegerValueType
        
        def patched_update_integer_value(self, value_type: Grid2opUpdateIntegerValueType, 
                                          value: np.ndarray, changed: np.ndarray) -> None:
            """Patched version: converts 0 to -1 instead of marking as unchanged.
            
            This fixes an issue where zero values were incorrectly handled,
            causing topology changes to not be applied properly.
            """
            value = value.copy()  # Don't modify original array
            value[value == 0] = -1
            _pypowsybl.update_grid2op_integer_value(self._handle, value_type, value, changed)
        
        PyPowSyBlBackend.update_integer_value = patched_update_integer_value
        return True, "Runtime patch applied successfully"
        
    except ImportError as e:
        return False, f"Cannot import pypowsybl2grid: {e}"
    except Exception as e:
        return False, f"Error applying patch: {e}"


def main():
    parser = argparse.ArgumentParser(description="Patch pypowsybl2grid backend")
    parser.add_argument("--verify", action="store_true", 
                        help="Only verify patch status, don't apply")
    args = parser.parse_args()
    
    print("=" * 60)
    print("pypowsybl2grid Backend Patch Tool")
    print("=" * 60)
    
    # Check current status
    needs_patch, status_msg = check_if_patch_needed()
    print(f"\nPatch status: {status_msg}")
    
    if args.verify:
        if needs_patch is True:
            print("⚠ Patch IS needed")
            sys.exit(1)
        elif needs_patch is False:
            print("✓ Patch NOT needed (already applied)")
            sys.exit(0)
        else:
            print("? Cannot determine patch status")
            sys.exit(2)
    
    # Apply patch if needed
    if needs_patch is True:
        print("\nApplying runtime patch...")
        success, msg = apply_runtime_patch()
        if success:
            print(f"✓ {msg}")
            sys.exit(0)
        else:
            print(f"✗ {msg}")
            sys.exit(1)
    elif needs_patch is False:
        print("\n✓ No action needed")
        sys.exit(0)
    else:
        print("\n⚠ Attempting runtime patch anyway...")
        success, msg = apply_runtime_patch()
        if success:
            print(f"✓ {msg}")
            sys.exit(0)
        else:
            print(f"✗ {msg}")
            sys.exit(1)


if __name__ == "__main__":
    main()
