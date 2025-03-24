from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

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