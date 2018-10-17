from copy import deepcopy, copy
from abc import ABCMeta, abstractmethod
import os
import os.path as op
import pydicom
import numpy as np
from arcana.node import Node
from arcana.data.file_format import FileFormat, Converter
from banana.interfaces.mrtrix import MRConvert
from banana.requirement import (
    dcm2niix_req, mrtrix3_req)
from banana.interfaces.converters import Dcm2niix  # @UnusedImport
import nibabel
# Import base file formats from Arcana for convenience
from arcana.data.file_format.standard import (
    text_format, directory_format, zip_format, targz_format,
    UnzipConverter, UnTarGzConverter)  # @UnusedImport


class ImageFileFormat(FileFormat, ABCMeta):

    @abstractmethod
    def load_header(self, fileset):
        pass

    @abstractmethod
    def load_array(self, fileset):
        pass

    def get_vox_sizes(self, fileset):
        return self.load_header(fileset)[self.VOX_SIZE_FIELD][1:4]


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
# Custom loader functions for different image types
# =====================================================================


def nifti_array_loader(path):
    return nibabel.load(path).get_data()


def nifti_header_loader(path):
    return dict(nibabel.load(path).get_header())


def dicom_array_loader(path):
    dcm_files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    image_stack = []
    for fname in dcm_files:
        image_stack.append(pydicom.dcmread(op.join(path, fname)).pixel_array)
    return np.asarray(image_stack)


def dicom_header_loader(path):
    dcm_files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    return pydicom.dcmread(op.join(path, dcm_files[0]))


# =====================================================================
# All Data Formats
# =====================================================================


# NeuroImaging data formats
dicom_format = FileFormat(name='dicom', extension=None,
                          directory=True, within_dir_exts=['.dcm'],
                          alternate_names=['secondary'],
                          array_loader=dicom_array_loader,
                          header_loader=dicom_header_loader)
nifti_format = FileFormat(name='nifti', extension='.nii',
                          converters={'dicom': Dcm2niixConverter,
                                      'analyze': MrtrixConverter,
                                      'nifti_gz': MrtrixConverter,
                                      'mrtrix': MrtrixConverter},
                          array_loader=nifti_array_loader,
                          header_loader=nifti_header_loader)
nifti_gz_format = FileFormat(name='nifti_gz', extension='.nii.gz',
                             converters={'dicom': Dcm2niixConverter,
                                         'nifti': MrtrixConverter,
                                         'analyze': MrtrixConverter,
                                         'mrtrix': MrtrixConverter},
                             array_loader=nifti_array_loader,
                             header_loader=nifti_header_loader)
analyze_format = FileFormat(name='analyze', extension='.img',
                            converters={'dicom': MrtrixConverter,
                                        'nifti': MrtrixConverter,
                                        'nifti_gz': MrtrixConverter,
                                        'mrtrix': MrtrixConverter})
mrtrix_format = FileFormat(name='mrtrix', extension='.mif',
                           converters={'dicom': MrtrixConverter,
                                       'nifti': MrtrixConverter,
                                       'nifti_gz': MrtrixConverter,
                                       'analyze': MrtrixConverter})

STD_IMAGE_FORMATS = [dicom_format, nifti_format, nifti_gz_format,
                     analyze_format, mrtrix_format]

multi_nifti_gz_format = FileFormat(name='multi_nifti_gz', extension=None,
                                   directory=True, within_dir_exts=['.nii.gz'],
                                   converters={'zip': UnzipConverter,
                                               'targz': UnTarGzConverter})

# Tractography formats
mrtrix_track_format = FileFormat(name='mrtrix_track', extension='.tck')

# Tabular formats
rfile_format = FileFormat(name='rdata', extension='.RData')
# matlab_format = FileFormat(name='matlab', extension='.mat')
csv_format = FileFormat(name='comma-separated_file', extension='.csv')
text_matrix_format = FileFormat(name='text_matrix', extension='.mat')

# Diffusion gradient-table data formats
fsl_bvecs_format = FileFormat(name='fsl_bvecs', extension='.bvec')
fsl_bvals_format = FileFormat(name='fsl_bvals', extension='.bval')
mrtrix_grad_format = FileFormat(name='mrtrix_grad', extension='.b')

# Tool-specific formats
eddy_par_format = FileFormat(name='eddy_par',
                             extension='.eddy_parameters')
freesurfer_recon_all_format = FileFormat(name='fs_recon_all',
                                         directory=True)
ica_format = FileFormat(name='ica', extension='.ica', directory=True)
par_format = FileFormat(name='parameters', extension='.par')
motion_mats_format = FileFormat(
    name='motion_mats', directory=True, within_dir_exts=['.mat'],
    desc=("Format used for storing motion matrices produced during "
          "motion detection pipeline"))


# General image formats
gif_format = FileFormat(name='gif', extension='.gif')
png_format = FileFormat(name='png', extension='.png')
jpg_format = FileFormat(name='jpg', extension='.jpg')

# PET formats
list_mode_format = FileFormat(name='pet_list_mode', extension='.bf')

# Raw formats
dat_format = FileFormat(name='dat', extension='.dat')

# MRS format
rda_format = FileFormat(name='raw', extension='.rda')

# Record list of all data formats registered by module (not really
# used currently but could be useful in future)
registered_file_formats = []

# Register all data formats in module
for file_format in copy(globals()).values():
    if isinstance(file_format, FileFormat):
        FileFormat.register(file_format)
        registered_file_formats.append(file_format.name)

# Since the conversion from DICOM->NIfTI is unfortunately slightly
# different between MRConvert and Dcm2niix, these data formats can
# be used in pipeline input specs that need to use MRConvert instead
# of Dcm2niix (i.e. motion-detection pipeline)
mrconvert_nifti_format = deepcopy(nifti_format)
mrconvert_nifti_format._converters['dicom'] = MrtrixConverter
mrconvert_nifti_gz_format = deepcopy(nifti_gz_format)
mrconvert_nifti_gz_format._converters['dicom'] = MrtrixConverter
