from .package_info import __version__
import os
# Ensure all data_formats are registered with Arcana
from .data_format import registered_data_formats

# Should be set explicitly in all FSL interfaces, but this squashes the warning
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
