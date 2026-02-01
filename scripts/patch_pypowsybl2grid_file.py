#!/usr/bin/env python3
"""
Script to patch pypowsybl2grid backend.py file directly.

This patches the installed pypowsybl2grid package to fix the zero-value handling
issue in update_integer_value method.

Usage:
    python scripts/patch_pypowsybl2grid_file.py [--dry-run] [--revert]
    
Options:
    --dry-run    Show what would be changed without modifying files
    --revert     Revert the patch (restore original behavior)
"""

import sys
import os
import argparse
import re
from pathlib import Path


def find_backend_file():
    """Find the pypowsybl backend.py file (in pypowsybl/grid2op/impl/)."""
    try:
        import pypowsybl
        package_dir = Path(pypowsybl.__file__).parent
        
        # The backend is in pypowsybl/grid2op/impl/backend.py
        backend_file = package_dir / "grid2op" / "impl" / "backend.py"
        
        if backend_file.exists():
            return backend_file
        
        # Fallback: try pypowsybl2grid package location
        try:
            import pypowsybl2grid
            pypowsybl2grid_dir = Path(pypowsybl2grid.__file__).parent
            backend_file = pypowsybl2grid_dir / "backend.py"
            if backend_file.exists():
                return backend_file
        except ImportError:
            pass
                
        print(f"Could not find backend.py in {package_dir}/grid2op/impl/")
        return None
        
    except ImportError:
        print("pypowsybl is not installed")
        return None


def read_file(filepath):
    """Read file contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# Pattern to find the original code (the direct pass-through)
ORIGINAL_PATTERN = r'(def update_integer_value\(self, value_type: Grid2opUpdateIntegerValueType, value: np\.ndarray, changed: np\.ndarray\) -> None:\n)(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))'

# The patched replacement - adds value[value==0] = -1 before the call
PATCHED_CODE = r'''\1\2# PATCHED: Convert 0 values to -1 for proper topology handling
\2value[value==0] = -1
\2\3'''

# Pattern to detect if already patched
PATCH_MARKER = '# PATCHED: Convert 0 values to -1'

# Pattern to revert the patch
REVERT_PATTERN = r'(def update_integer_value\(self, value_type: Grid2opUpdateIntegerValueType, value: np\.ndarray, changed: np\.ndarray\) -> None:\n)(\s*)# PATCHED: Convert 0 values to -1 for proper topology handling\n\s*value\[value==0\] = -1\n(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))'
ORIGINAL_CODE = r'\1\3\4'


def is_patched(content):
    """Check if the file is already patched."""
    return PATCH_MARKER in content


def apply_patch(content):
    """Apply the patch to the content."""
    if is_patched(content):
        return content, False, "Already patched"
    
    # Check if the original pattern exists
    match = re.search(ORIGINAL_PATTERN, content)
    if not match:
        # Debug: show what we're looking for
        if 'def update_integer_value' in content:
            # Find the actual method signature for debugging
            import re as re_debug
            method_match = re_debug.search(r'def update_integer_value[^:]+:[^\n]*\n[^\n]+', content)
            if method_match:
                print(f"Found method but pattern didn't match. Actual code:")
                print(f"  {method_match.group()[:200]}...")
        return content, False, "Original pattern not found - file may have different version"
    
    # Apply the patch
    new_content = re.sub(ORIGINAL_PATTERN, PATCHED_CODE, content)
    
    if new_content == content:
        return content, False, "Regex replacement failed"
    
    return new_content, True, "Patch applied successfully"


def revert_patch(content):
    """Revert the patch."""
    if not is_patched(content):
        return content, False, "Not patched - nothing to revert"
    
    # Revert the patch
    new_content = re.sub(REVERT_PATTERN, ORIGINAL_CODE, content)
    
    if new_content == content:
        return content, False, "Revert regex failed"
    
    return new_content, True, "Patch reverted successfully"


def main():
    parser = argparse.ArgumentParser(description="Patch pypowsybl2grid backend.py")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Show what would be changed without modifying files")
    parser.add_argument("--revert", action="store_true",
                        help="Revert the patch")
    args = parser.parse_args()
    
    print("=" * 60)
    print("pypowsybl2grid Backend File Patcher")
    print("=" * 60)
    
    # Find the backend file
    backend_file = find_backend_file()
    if not backend_file:
        print("\n✗ Could not locate backend.py file")
        sys.exit(1)
    
    print(f"\nFound backend file: {backend_file}")
    
    # Read current content
    content = read_file(backend_file)
    
    # Check current state
    patched = is_patched(content)
    print(f"Current state: {'PATCHED' if patched else 'ORIGINAL'}")
    
    if args.revert:
        new_content, changed, message = revert_patch(content)
    else:
        new_content, changed, message = apply_patch(content)
    
    print(f"\nResult: {message}")
    
    if changed:
        if args.dry_run:
            print("\n[DRY RUN] Would modify file. Changes:")
            print("-" * 40)
            # Show diff-like output
            old_lines = content.split('\n')
            new_lines = new_content.split('\n')
            for i, (old, new) in enumerate(zip(old_lines, new_lines)):
                if old != new:
                    print(f"Line {i+1}:")
                    print(f"  - {old}")
                    print(f"  + {new}")
            print("-" * 40)
            print("\nRun without --dry-run to apply changes.")
        else:
            # Actually write the file
            write_file(backend_file, new_content)
            print(f"\n✓ File modified: {backend_file}")
            
            # Verify
            verify_content = read_file(backend_file)
            if args.revert:
                if not is_patched(verify_content):
                    print("✓ Patch successfully reverted")
                    sys.exit(0)
                else:
                    print("✗ Verification failed - file still appears patched")
                    sys.exit(1)
            else:
                if is_patched(verify_content):
                    print("✓ Patch successfully applied")
                    sys.exit(0)
                else:
                    print("✗ Verification failed - patch marker not found")
                    sys.exit(1)
    else:
        if args.revert and not patched:
            print("\n✓ File is already in original state")
            sys.exit(0)
        elif not args.revert and patched:
            print("\n✓ File is already patched")
            sys.exit(0)
        else:
            print("\n✗ No changes made")
            sys.exit(1)


if __name__ == "__main__":
    main()
