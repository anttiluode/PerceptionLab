#!/usr/bin/env python3
"""
Auto Dependency Installer for Antti's Perception Laboratory
Scans host and node files for import statements and installs missing packages.

Usage:
    python install_dependencies.py
"""

import os
import sys
import ast
import subprocess
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Mapping of import names to pip package names (when they differ)
IMPORT_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'PyQt6': 'PyQt6',
    'pyqtgraph': 'pyqtgraph',
    'pyaudio': 'pyaudio',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'mne': 'mne',
    'pywt': 'PyWavelets',
    'numba': 'numba',
    'torch': 'torch',
    'scipy': 'scipy',
    'networkx': 'networkx',
}

# Known standard library modules (don't try to install these)
STDLIB_MODULES = {
    'sys', 'os', 'time', 'math', 'random', 'json', 'inspect', 
    'importlib', 'collections', 'pathlib', 'subprocess', 'ast',
    're', 'typing', 'io', 'warnings', 'copy', 'pickle', 'datetime',
    'functools', 'itertools', 'operator', 'string', 'tempfile',
    'threading', 'multiprocessing', '__main__', '__future__',
}

class DependencyScanner:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.all_imports: Set[str] = set()
        self.file_imports: Dict[str, Set[str]] = {}
        self.installed_packages: Set[str] = self._get_installed_packages()
        
    def _get_installed_packages(self) -> Set[str]:
        """Get list of currently installed packages."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                capture_output=True,
                text=True,
                check=True
            )
            packages = set()
            for line in result.stdout.split('\n'):
                if '==' in line:
                    pkg_name = line.split('==')[0].lower()
                    packages.add(pkg_name)
            return packages
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get installed packages: {e}")
            return set()
    
    def _extract_imports_from_file(self, filepath: str) -> Set[str]:
        """Extract all import statements from a Python file using AST."""
        imports = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse with AST
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            # Get the top-level package name
                            pkg = alias.name.split('.')[0]
                            imports.add(pkg)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            pkg = node.module.split('.')[0]
                            imports.add(pkg)
            except SyntaxError:
                # If AST parsing fails, fall back to regex
                if self.verbose:
                    print(f"  Warning: AST parsing failed for {filepath}, using regex fallback")
                imports.update(self._extract_imports_regex(content))
                
        except Exception as e:
            if self.verbose:
                print(f"  Error reading {filepath}: {e}")
        
        return imports
    
    def _extract_imports_regex(self, content: str) -> Set[str]:
        """Fallback method using regex to find imports."""
        imports = set()
        
        # Match: import xyz
        for match in re.finditer(r'^\s*import\s+([a-zA-Z0-9_]+)', content, re.MULTILINE):
            imports.add(match.group(1))
        
        # Match: from xyz import
        for match in re.finditer(r'^\s*from\s+([a-zA-Z0-9_]+)\s+import', content, re.MULTILINE):
            imports.add(match.group(1))
        
        return imports
    
    def scan_file(self, filepath: str) -> Set[str]:
        """Scan a single Python file for imports."""
        if self.verbose:
            print(f"Scanning: {filepath}")
        
        imports = self._extract_imports_from_file(filepath)
        self.file_imports[filepath] = imports
        self.all_imports.update(imports)
        
        return imports
    
    def scan_directory(self, directory: str, pattern: str = "*.py") -> None:
        """Recursively scan a directory for Python files."""
        path = Path(directory)
        
        if not path.exists():
            if self.verbose:
                print(f"Warning: Directory '{directory}' does not exist")
            return
        
        for py_file in path.rglob(pattern):
            if py_file.is_file():
                self.scan_file(str(py_file))
    
    def get_missing_packages(self) -> Dict[str, str]:
        """
        Get packages that need to be installed.
        Returns dict: {import_name: pip_package_name}
        """
        missing = {}
        
        for import_name in self.all_imports:
            # Skip standard library
            if import_name in STDLIB_MODULES:
                continue
            
            # Get the pip package name
            pip_name = IMPORT_TO_PACKAGE.get(import_name, import_name)
            
            # Check if installed (case-insensitive)
            if pip_name.lower() not in self.installed_packages:
                missing[import_name] = pip_name
        
        return missing
    
    def install_packages(self, packages: List[str], dry_run: bool = False) -> Tuple[List[str], List[str]]:
        """
        Install packages using pip.
        Returns (successful, failed) lists.
        """
        successful = []
        failed = []
        
        for package in packages:
            if dry_run:
                print(f"[DRY RUN] Would install: {package}")
                successful.append(package)
                continue
            
            print(f"\nInstalling {package}...")
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✓ Successfully installed {package}")
                successful.append(package)
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}")
                if self.verbose:
                    print(f"  Error: {e.stderr}")
                failed.append(package)
        
        return successful, failed
    
    def print_summary(self):
        """Print a summary of found imports."""
        print("\n" + "="*60)
        print("DEPENDENCY SCAN SUMMARY")
        print("="*60)
        print(f"\nTotal Python files scanned: {len(self.file_imports)}")
        print(f"Total unique imports found: {len(self.all_imports)}")
        
        missing = self.get_missing_packages()
        if missing:
            print(f"\n⚠ Missing packages: {len(missing)}")
            for import_name, pip_name in sorted(missing.items()):
                status = "✗ NOT INSTALLED"
                if import_name != pip_name:
                    print(f"  {status}: {import_name} (install via: {pip_name})")
                else:
                    print(f"  {status}: {import_name}")
        else:
            print("\n✓ All required packages are installed!")
        
        print("\n" + "="*60)


def main():
    print("="*60)
    print("Antti's Perception Laboratory - Dependency Installer")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('perception_lab_host.py'):
        print("\n⚠ Warning: perception_lab_host.py not found in current directory")
        print("Please run this script from the Perception Lab root directory")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    scanner = DependencyScanner(verbose=True)
    
    # Scan the host file
    print("\n📄 Scanning host file...")
    if os.path.exists('perception_lab_host.py'):
        scanner.scan_file('perception_lab_host.py')
    else:
        print("  Host file not found, skipping...")
    
    # Scan the nodes directory
    print("\n📁 Scanning nodes directory...")
    if os.path.exists('nodes'):
        scanner.scan_directory('nodes')
    else:
        print("  Nodes directory not found, skipping...")
    
    # Print summary
    scanner.print_summary()
    
    # Get missing packages
    missing = scanner.get_missing_packages()
    
    if not missing:
        print("\n🎉 All dependencies are already installed!")
        return 0
    
    # Ask user if they want to install
    print(f"\n{len(missing)} package(s) need to be installed.")
    print("\nOptions:")
    print("  1. Install all missing packages")
    print("  2. Install packages one by one (with confirmation)")
    print("  3. Show install commands only (no installation)")
    print("  4. Exit without installing")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Install all
        print("\n" + "="*60)
        print("INSTALLING ALL PACKAGES")
        print("="*60)
        packages = list(missing.values())
        successful, failed = scanner.install_packages(packages)
        
        print("\n" + "="*60)
        print(f"Installation complete: {len(successful)} succeeded, {len(failed)} failed")
        if failed:
            print(f"\nFailed packages: {', '.join(failed)}")
            print("You may need to install these manually.")
        print("="*60)
        
    elif choice == '2':
        # Install with confirmation
        successful = []
        failed = []
        skipped = []
        
        for import_name, pip_name in sorted(missing.items()):
            prompt = f"\nInstall {pip_name}"
            if import_name != pip_name:
                prompt += f" (for {import_name})"
            prompt += "? (y/n/q): "
            
            response = input(prompt).strip().lower()
            
            if response == 'q':
                print("Installation cancelled by user")
                break
            elif response == 'y':
                succ, fail = scanner.install_packages([pip_name])
                successful.extend(succ)
                failed.extend(fail)
            else:
                skipped.append(pip_name)
        
        print("\n" + "="*60)
        print(f"Installed: {len(successful)}, Failed: {len(failed)}, Skipped: {len(skipped)}")
        print("="*60)
        
    elif choice == '3':
        # Show commands only
        print("\n" + "="*60)
        print("INSTALLATION COMMANDS")
        print("="*60)
        print("\nCopy and paste these commands to install manually:\n")
        
        for pip_name in sorted(set(missing.values())):
            print(f"pip install {pip_name}")
        
        print("\nOr install all at once:")
        print(f"pip install {' '.join(sorted(set(missing.values())))}")
        print("\n" + "="*60)
        
    else:
        print("\nExiting without installing.")
        return 0
    
    # Final check
    print("\n\n📊 Running final dependency check...")
    scanner_final = DependencyScanner(verbose=False)
    if os.path.exists('perception_lab_host.py'):
        scanner_final.scan_file('perception_lab_host.py')
    if os.path.exists('nodes'):
        scanner_final.scan_directory('nodes')
    
    missing_final = scanner_final.get_missing_packages()
    
    if missing_final:
        print(f"\n⚠ {len(missing_final)} package(s) still missing:")
        for import_name, pip_name in sorted(missing_final.items()):
            print(f"  - {pip_name}")
        print("\nSome packages may require system dependencies or special installation.")
        print("Check the documentation for:")
        if 'pyaudio' in [p.lower() for p in missing_final.values()]:
            print("  - PyAudio: May need portaudio (brew install portaudio / apt-get install portaudio19-dev)")
        if 'torch' in [p.lower() for p in missing_final.values()]:
            print("  - PyTorch: Visit pytorch.org for installation instructions")
    else:
        print("\n✓ All dependencies are now installed!")
        print("\nYou can now run: python perception_lab_host.py")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)