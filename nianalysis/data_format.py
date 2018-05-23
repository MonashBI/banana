from copy import deepcopy, copy
from arcana.node import Node
from arcana.data_format import DataFormat, Converter
from arcana.interfaces.mrtrix import MRConvert
from nianalysis.requirement import (
    dcm2niix_req, mrtrix3_req)
from arcana.interfaces.converters import Dcm2niix
from arcana.data_format import (
    text_format, directory_format, zip_format, targz_format)  # @UnusedImport


class Dcm2niixConverter(Converter):

    requirements = [dcm2niix_req]

    def get_node(self, name):
        convert_node = Node(Dcm2niix(), name=name,
                            requirements=self.requirements,
                            wall_time=20)
        convert_node.inputs.compression = 'y'
        return convert_node, 'input_dir', 'converted'


class MrtrixConverter(Converter):

    requirements = [mrtrix3_req]

    def get_node(self, name):
        convert_node = Node(MRConvert(), name=name,
                            requirements=self.requirements)
        convert_node.inputs.out_ext = self._output_format.extension
        convert_node.inputs.quiet = True
        return convert_node, 'in_file', 'out_file'


# =====================================================================
# All Data Formats
# =====================================================================


# NeuroImaging data formats
dicom_format = DataFormat(name='dicom', extension=None,
                          directory=True, within_dir_exts=['.dcm'])
nifti_format = DataFormat(name='nifti', extension='.nii',
                          converters={'dicom': Dcm2niixConverter,
                                      'analyze': MrtrixConverter,
                                      'nifti_gz': MrtrixConverter,
                                      'mrtrix': MrtrixConverter})
nifti_gz_format = DataFormat(name='nifti_gz', extension='.nii.gz',
                             converters={'dicom': Dcm2niixConverter,
                                         'nifti': MrtrixConverter,
                                         'analyze': MrtrixConverter,
                                         'mrtrix': MrtrixConverter})
analyze_format = DataFormat(name='analyze', extension='.img',
                            converters={'dicom': MrtrixConverter,
                                        'nifti': MrtrixConverter,
                                        'nifti_gz': MrtrixConverter,
                                        'mrtrix': MrtrixConverter})
mrtrix_format = DataFormat(name='mrtrix', extension='.mif',
                           converters={'dicom': MrtrixConverter,
                                       'nifti': MrtrixConverter,
                                       'nifti_gz': MrtrixConverter,
                                       'analyze': MrtrixConverter})

# Tabular formats
rdata_format = DataFormat(name='rdata', extension='.RData')
# matlab_format = DataFormat(name='matlab', extension='.mat')
csv_format = DataFormat(name='comma-separated_file', extension='.csv')
text_matrix_format = DataFormat(name='text_matrix', extension='.mat')

# Diffusion gradient-table data formats
fsl_bvecs_format = DataFormat(name='fsl_bvecs', extension='.bvec')
fsl_bvals_format = DataFormat(name='fsl_bvals', extension='.bval')
mrtrix_grad_format = DataFormat(name='mrtrix_grad', extension='.b')

# Tool-specific formats
eddy_par_format = DataFormat(name='eddy_par',
                             extension='.eddy_parameters')
freesurfer_recon_all_format = DataFormat(name='fs_recon_all',
                                         directory=True)
ica_format = DataFormat(name='ica', extension='.ica', directory=True)
par_format = DataFormat(name='parameters', extension='.par')
motion_mats_format = DataFormat(
    name='motion_mats', directory=True, within_dir_exts=['.mat'],
    desc=("Format used for storing motion matrices produced during "
          "motion detection pipeline"))


# General image formats
gif_format = DataFormat(name='gif', extension='.gif')
png_format = DataFormat(name='portable-network-graphics',
                        extension='.png')

# PET formats
list_mode_format = DataFormat(name='pet_list_mode', extension='.bf')

# Raw formats
dat_format = DataFormat(name='dat', extension='.dat')

# Record list of all data formats registered by module (not really
# used currently but could be useful in future)
registered_data_formats = []

# Register all data formats in module
for data_format in copy(globals()).itervalues():
    if isinstance(data_format, DataFormat):
        DataFormat.register(data_format)
        registered_data_formats.append(data_format.name)

# Since the conversion from DICOM->NIfTI is unfortunately slightly
# different between MRConvert and Dcm2niix, these data formats can
# be used in pipeline input specs that need to use MRConvert instead
# of Dcm2niix (i.e. motion-detection pipeline)
mrconvert_nifti_format = deepcopy(nifti_format)
mrconvert_nifti_format._converters['dicom'] = MrtrixConverter
mrconvert_nifti_gz_format = deepcopy(nifti_gz_format)
mrconvert_nifti_gz_format._converters['dicom'] = MrtrixConverter
