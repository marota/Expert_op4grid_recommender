#!/usr/bin/env python3
"""
Script to patch pypowsybl2grid backend.py file directly.

This patches the installed pypowsybl2grid package to fix the zero-value handling
issue in update_integer_value method.

Usage:
    python scripts/patch_pypowsybl2grid_file.py [--dry-run] [--revert] [--debug]

Options:
    --dry-run    Show what would be changed without modifying files
    --revert     Revert the patch (restore original behavior)
    --debug      Show detailed debugging information
"""

import sys
import os
import argparse
import re
from pathlib import Path


def find_backend_file():
    """Find the pypowsybl backend.py file (in pypowsybl/grid2op/impl/)."""
    locations_tried = []

    # Try pypowsybl package first
    try:
        import pypowsybl
        package_dir = Path(pypowsybl.__file__).parent
        print(f"pypowsybl version: {getattr(pypowsybl, '__version__', 'unknown')}")
        print(f"pypowsybl location: {package_dir}")

        # The backend is in pypowsybl/grid2op/impl/backend.py
        backend_file = package_dir / "grid2op" / "impl" / "backend.py"
        locations_tried.append(str(backend_file))

        if backend_file.exists():
            return backend_file

    except ImportError:
        print("pypowsybl is not installed")

    # Fallback: try pypowsybl2grid package location
    try:
        import pypowsybl2grid
        pypowsybl2grid_dir = Path(pypowsybl2grid.__file__).parent
        print(f"pypowsybl2grid version: {getattr(pypowsybl2grid, '__version__', 'unknown')}")
        print(f"pypowsybl2grid location: {pypowsybl2grid_dir}")

        # Try multiple possible locations
        possible_files = [
            pypowsybl2grid_dir / "backend.py",
            pypowsybl2grid_dir / "pypowsybl_backend.py",
        ]

        for backend_file in possible_files:
            locations_tried.append(str(backend_file))
            if backend_file.exists():
                return backend_file

    except ImportError:
        print("pypowsybl2grid is not installed")

    print(f"Could not find backend.py. Locations tried:")
    for loc in locations_tried:
        print(f"  - {loc}")
    return None


def read_file(filepath):
    """Read file contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# Pattern to detect if already patched
PATCH_MARKER = '# PATCHED: Convert 0 values to -1'

# Multiple patterns to try for different versions
ORIGINAL_PATTERNS = [
    # Pattern 1: Standard format with type hints
    r'(def update_integer_value\(self, value_type: Grid2opUpdateIntegerValueType, value: np\.ndarray, changed: np\.ndarray\) -> None:\n)(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))',
    # Pattern 2: Without return type hint
    r'(def update_integer_value\(self, value_type: Grid2opUpdateIntegerValueType, value: np\.ndarray, changed: np\.ndarray\):\n)(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))',
    # Pattern 3: With different parameter type hints (npt.NDArray)
    r'(def update_integer_value\(self, value_type: Grid2opUpdateIntegerValueType, value: npt\.NDArray, changed: npt\.NDArray\) -> None:\n)(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))',
    # Pattern 4: Generic typing
    r'(def update_integer_value\(self, value_type:\s*\w+,\s*value:\s*[\w\.\[\]]+,\s*changed:\s*[\w\.\[\]]+\)[^:]*:\n)(\s*)(_pypowsybl\.update_grid2op_integer_value\(self\._handle, value_type, value, changed\))',
]

# The patched replacement - adds value[value==0] = -1 before the call
PATCHED_CODE = r'''\1\2# PATCHED: Convert 0 values to -1 for proper topology handling
\2value[value==0] = -1
\2\3'''


def is_patched(content):
    """Check if the file is already patched."""
    return PATCH_MARKER in content


def debug_file_content(content):
    """Print debug information about the file content."""
    print("\n" + "=" * 60)
    print("DEBUG: File Content Analysis")
    print("=" * 60)

    # Check for key strings
    checks = [
        ('update_integer_value', 'def update_integer_value' in content),
        ('_pypowsybl', '_pypowsybl' in content),
        ('Grid2opUpdateIntegerValueType', 'Grid2opUpdateIntegerValueType' in content),
        ('np.ndarray', 'np.ndarray' in content),
        ('npt.NDArray', 'npt.NDArray' in content),
        ('update_grid2op_integer_value', 'update_grid2op_integer_value' in content),
    ]

    print("\nKey string presence:")
    for name, found in checks:
        status = "FOUND" if found else "NOT FOUND"
        print(f"  {name}: {status}")

    # Find and show the update_integer_value method
    if 'def update_integer_value' in content:
        print("\nSearching for update_integer_value method...")

        # Find all occurrences
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def update_integer_value' in line:
                print(f"\nFound at line {i+1}:")
                # Show context: 2 lines before and 5 lines after
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    # Show repr to see exact characters including whitespace
                    print(f"  {marker} L{j+1}: {repr(lines[j])}")
    else:
        print("\n'def update_integer_value' not found in file!")

        # Show what methods ARE in the file
        print("\nMethods found in file:")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                print(f"  L{i+1}: {line.strip()[:80]}")

    print("=" * 60 + "\n")


def apply_patch(content, debug=False):
    """Apply the patch to the content."""
    if is_patched(content):
        return content, False, "Already patched"

    if debug:
        debug_file_content(content)

    # Try each pattern
    for i, pattern in enumerate(ORIGINAL_PATTERNS):
        match = re.search(pattern, content)
        if match:
            if debug:
                print(f"Pattern {i+1} matched!")
                print(f"  Match groups: {len(match.groups())}")
                for j, g in enumerate(match.groups()):
                    print(f"  Group {j+1}: {repr(g[:50])}...")

            # Apply the patch
            new_content = re.sub(pattern, PATCHED_CODE, content)

            if new_content != content:
                return new_content, True, f"Patch applied successfully (pattern {i+1})"
            else:
                if debug:
                    print(f"  Pattern matched but substitution had no effect!")

    # If no pattern matched, try a line-by-line approach as fallback
    print("\nTrying line-by-line fallback approach...")

    lines = content.split('\n')
    new_lines = []
    patched = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for the method definition
        if 'def update_integer_value' in line and not patched:
            new_lines.append(line)
            i += 1

            # Skip any docstrings or comments
            while i < len(lines) and (lines[i].strip().startswith('"""') or
                                       lines[i].strip().startswith("'''") or
                                       lines[i].strip().startswith('#')):
                new_lines.append(lines[i])
                i += 1

            # Next should be the implementation line
            if i < len(lines) and '_pypowsybl.update_grid2op_integer_value' in lines[i]:
                impl_line = lines[i]
                # Get the indentation from the implementation line
                indent = len(impl_line) - len(impl_line.lstrip())
                indent_str = impl_line[:indent]

                # Add our patch lines
                new_lines.append(f"{indent_str}# PATCHED: Convert 0 values to -1 for proper topology handling")
                new_lines.append(f"{indent_str}value[value==0] = -1")
                new_lines.append(impl_line)
                patched = True
                i += 1
                continue

        new_lines.append(line)
        i += 1

    if patched:
        return '\n'.join(new_lines), True, "Patch applied successfully (line-by-line fallback)"

    # Debug output if we still couldn't patch
    if not debug:
        # Show debug info even if --debug wasn't specified
        debug_file_content(content)

    return content, False, "Could not find pattern to patch - see debug output above"


def revert_patch(content, debug=False):
    """Revert the patch."""
    if not is_patched(content):
        return content, False, "Not patched - nothing to revert"

    # Try to revert by removing our added lines
    lines = content.split('\n')
    new_lines = []
    i = 0
    reverted = False

    while i < len(lines):
        line = lines[i]

        # Skip our patch marker and the following line
        if PATCH_MARKER in line:
            # Skip this comment line
            i += 1
            # Skip the value[value==0] = -1 line if it follows
            if i < len(lines) and 'value[value==0] = -1' in lines[i]:
                i += 1
                reverted = True
                continue

        new_lines.append(line)
        i += 1

    if reverted:
        return '\n'.join(new_lines), True, "Patch reverted successfully"

    return content, False, "Could not find patch to revert"


def main():
    parser = argparse.ArgumentParser(description="Patch pypowsybl2grid backend.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without modifying files")
    parser.add_argument("--revert", action="store_true",
                        help="Revert the patch")
    parser.add_argument("--debug", action="store_true",
                        help="Show detailed debugging information")
    args = parser.parse_args()

    print("=" * 60)
    print("pypowsybl2grid Backend File Patcher")
    print("=" * 60)

    # Find the backend file
    backend_file = find_backend_file()
    if not backend_file:
        print("\n[X] Could not locate backend.py file")
        sys.exit(1)

    print(f"\nFound backend file: {backend_file}")
    print(f"File size: {backend_file.stat().st_size} bytes")

    # Read current content
    content = read_file(backend_file)
    print(f"Content length: {len(content)} characters")

    # Check current state
    patched = is_patched(content)
    print(f"Current state: {'PATCHED' if patched else 'ORIGINAL'}")

    if args.revert:
        new_content, changed, message = revert_patch(content, debug=args.debug)
    else:
        new_content, changed, message = apply_patch(content, debug=args.debug)

    print(f"\nResult: {message}")

    if changed:
        if args.dry_run:
            print("\n[DRY RUN] Would modify file. Changes:")
            print("-" * 40)
            # Show diff-like output
            old_lines = content.split('\n')
            new_lines = new_content.split('\n')

            # Use a simple diff approach
            import difflib
            diff = difflib.unified_diff(old_lines, new_lines, lineterm='', n=3)
            for line in diff:
                print(line)
            print("-" * 40)
            print("\nRun without --dry-run to apply changes.")
        else:
            # Actually write the file
            write_file(backend_file, new_content)
            print(f"\n[OK] File modified: {backend_file}")

            # Verify
            verify_content = read_file(backend_file)
            if args.revert:
                if not is_patched(verify_content):
                    print("[OK] Patch successfully reverted")
                    sys.exit(0)
                else:
                    print("[X] Verification failed - file still appears patched")
                    sys.exit(1)
            else:
                if is_patched(verify_content):
                    print("[OK] Patch successfully applied")
                    sys.exit(0)
                else:
                    print("[X] Verification failed - patch marker not found")
                    sys.exit(1)
    else:
        if args.revert and not patched:
            print("\n[OK] File is already in original state")
            sys.exit(0)
        elif not args.revert and patched:
            print("\n[OK] File is already patched")
            sys.exit(0)
        else:
            print("\n[X] No changes made")
            sys.exit(1)


if __name__ == "__main__":
    main()
