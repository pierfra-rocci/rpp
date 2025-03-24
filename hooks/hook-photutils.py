from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, get_package_paths
import os
import glob

# Get all data files
datas = collect_data_files('photutils')
datas += copy_metadata('photutils')

# Get all submodules to make sure we include compiled extensions
hiddenimports = collect_submodules('photutils')

# Explicitly include any problematic modules
hiddenimports += [
    'photutils.geometry.core',
    'photutils.geometry.circular_overlap',
    'photutils.geometry.elliptical_overlap',
    'photutils.geometry.rectangular_overlap'
]

# Add binary files (.pyd files for Windows)
pkg_base, pkg_dir = get_package_paths('photutils')
pyd_files = glob.glob(os.path.join(pkg_dir, '**', '*.pyd'), recursive=True)
binaries = [(f, os.path.join('photutils', os.path.relpath(os.path.dirname(f), pkg_dir))) for f in pyd_files]