from copy import copy
from nianalysis.node import Node
from nianalysis.data_format import DataFormat, Converter
from nianalysis.interfaces.mrtrix import MRConvert
from mbianalysis.requirement import (
    dcm2niix_req, mrtrix3_req)
from nianalysis.interfaces.converters import Dcm2niix
from nianalysis.data_format import (
    text_format, directory_format, zip_format, targz_format)  # @UnusedImport
from nipype.utils.filemanip import split_filename


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
    

class Nii2DicomInputSpec(TraitedSpec):
    in_file = File(mandatory=True, desc='input nifti file')
    reference_dicom = traits.List(mandatory=True, desc='original umap')
#     out_file = Directory(genfile=True, desc='the output dicom file')


class Nii2DicomOutputSpec(TraitedSpec):
    out_file = Directory(exists=True, desc='the output dicom file')


class Nii2Dicom(BaseInterface):
    """
    Creates two umaps in dicom format

    fully compatible with the UTE study:

    Attenuation Correction pipeline

    """

    input_spec = Nii2DicomInputSpec
    output_spec = Nii2DicomOutputSpec

    def _run_interface(self, runtime):
        dcms = self.inputs.reference_dicom
        to_remove = [x for x in dcms if '.dcm' not in x]
        if to_remove:
            for f in to_remove:
                dcms.remove(f)
#         dcms = glob.glob(self.inputs.reference_dicom+'/*.dcm')
#         if not dcms:
#             dcms = glob.glob(self.inputs.reference_dicom+'/*.IMA')
#         if not dcms:
#             raise Exception('No DICOM files found in {}'
#                             .format(self.inputs.reference_dicom))
        nifti_image = nib.load(self.inputs.in_file)
        nii_data = nifti_image.get_data()
        if len(dcms) != nii_data.shape[2]:
            raise Exception('Different number of nifti and dicom files '
                            'provided. Dicom to nifti conversion require the '
                            'same number of files in order to run. Please '
                            'check.')
        os.mkdir('nifti2dicom')
        _, basename, _ = split_filename(self.inputs.in_file)
        for i in range(nii_data.shape[2]):
            dcm = pydicom.read_file(dcms[i])
            nifti = nii_data[:, :, i]
            nifti = nifti.astype('uint16')
            dcm.pixel_array.setflags(write=True)
            dcm.pixel_array.flat[:] = nifti.flat[:]
            dcm.PixelData = dcm.pixel_array.T.tostring()
            dcm.save_as('nifti2dicom/{0}_vol{1}.dcm'
                        .format(basename, str(i).zfill(4)))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = (
            os.getcwd()+'/nifti2dicom')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            fpath = self.inputs.out_file
        else:
            fname = (
                split_extension(os.path.basename(self.inputs.in_file))[0] +
                '_dicom')
            fpath = os.path.join(os.getcwd(), fname)
        return fpath


# =====================================================================
# All Data Formats
# =====================================================================


# NeuroImaging data formats
dicom_format = DataFormat(name='dicom', extension=None,
                          directory=True, within_dir_exts=['.dcm'],
                          converters={'nifti': Dcm2niixConverter,
                                      'nifti_gz': Dcm2niixConverter,
                                      'mrtrix': MrtrixConverter,
                                      'analyze': MrtrixConverter})
nifti_format = DataFormat(name='nifti', extension='.nii',
                          converters={'analyze': MrtrixConverter,
                                      'nifti_gz': MrtrixConverter,
                                      'mrtrix': MrtrixConverter})
nifti_gz_format = DataFormat(name='nifti_gz', extension='.nii.gz',
                             converters={'nifti': MrtrixConverter,
                                         'analyze': MrtrixConverter,
                                         'mrtrix': MrtrixConverter})
analyze_format = DataFormat(name='analyze', extension='.img',
                            converters={'nifti': MrtrixConverter,
                                        'nifti_gz': MrtrixConverter,
                                        'mrtrix': MrtrixConverter})
mrtrix_format = DataFormat(name='mrtrix', extension='.mif',
                           converters={'nifti': MrtrixConverter,
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


# General image formats
gif_format = DataFormat(name='gif', extension='.gif')
png_format = DataFormat(name='portable-network-graphics',
                        extension='.png')

# PET formats
list_mode_format = DataFormat(name='pet_list_mode', extension='.bf')

# Record list of all data formats registered by module (not really
# used currently but could be useful in future)
registered_data_formats = []

# Register all data formats in module
for data_format in copy(globals()).itervalues():
    if isinstance(data_format, DataFormat):
        DataFormat.register(data_format)
        registered_data_formats.append(data_format.name)
