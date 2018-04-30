from copy import copy
from nianalysis.nodes import Node
from nianalysis.data_formats import DataFormat, Converter
from nianalysis.interfaces.mrtrix import MRConvert
from nianalysis.exceptions import NiAnalysisUsageError
from nianalysis.interfaces.utils import (
    ZipDir, UnzipDir, TarGzDir, UnTarGzDir)
from mbianalysis.requirements import (
    dcm2niix_req, mricrogl_req, mrtrix3_req)
from nianalysis.interfaces.converters import Dcm2niix


nifti_format = DataFormat(name='nifti', extension='.nii')
nifti_gz_format = DataFormat(name='nifti_gz', extension='.nii.gz')
mrtrix_format = DataFormat(name='mrtrix', extension='.mif')
analyze_format = DataFormat(name='analyze', extension='.img')
dicom_format = DataFormat(name='dicom', extension=None,
                          directory=True, within_dir_exts=['.dcm'])
fsl_bvecs_format = DataFormat(name='fsl_bvecs', extension='.bvec')
fsl_bvals_format = DataFormat(name='fsl_bvals', extension='.bval')
mrtrix_grad_format = DataFormat(name='mrtrix_grad', extension='.b')
matlab_format = DataFormat(name='matlab', extension='.mat')
freesurfer_recon_all_format = DataFormat(
    name='fs_recon_all', extension=None, directory=True)
zip_format = DataFormat(name='zip', extension='.zip')
directory_format = DataFormat(name='directory', extension=None,
                              directory=True)
text_matrix_format = DataFormat(name='text_matrix', extension='.mat')
text_format = DataFormat(name='text', extension='.txt')
rdata_format = DataFormat(name='rdata', extension='.RData')
ica_format = DataFormat(name='ica', extension='.ica', directory=True)
par_format = DataFormat(name='parameters', extension='.par')
gif_format = DataFormat(name='gif', extension='.gif')
targz_format = DataFormat(name='targz', extension='.tar.gz')
csv_format = DataFormat(name='comma-separated_file', extension='.csv')
png_format = DataFormat(name='portable-network-graphics',
                        extension='.png')
eddy_par_format = DataFormat(name='eddy_par',
                             extension='.eddy_parameters')


class Dcm2niixConverter(Converter):

    requirements = [(dcm2niix_req, mricrogl_req)]

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(Dcm2niix(), name=node_name,
                            requirements=self.requirements, wall_time=20)
        convert_node.inputs.compression = 'y'
        return convert_node, 'input_dir', 'converted'

    def input_formats(self):
        return [dicom_format]

    def output_formats(self):
        return [nifti_format, nifti_gz_format]


class MrtrixConverter(Converter):

    requirements = [mrtrix3_req]

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(MRConvert(), name=node_name,
                            requirements=self.requirements)
        convert_node.inputs.out_ext = output_format.extension
        convert_node.inputs.quiet = True
        return convert_node, 'in_file', 'out_file'

    def input_formats(self):
        return [nifti_format, nifti_gz_format, mrtrix_format,
                analyze_format, dicom_format]

    def output_formats(self):
        return [nifti_format, nifti_gz_format, analyze_format,
                mrtrix_format]


class UnzipConverter(Converter):

    requirements = []

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(UnzipDir(), name=node_name,
                            memory=12000)
        return convert_node, 'zipped', 'unzipped'

    def input_formats(self):
        return [zip_format]

    def output_formats(self):
        return [directory_format]


class ZipConverter(Converter):

    requirements = []

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(ZipDir(), name=node_name,
                            memory=12000)
        return convert_node, 'dirname', 'zipped'

    def input_formats(self):
        return [directory_format]

    def output_formats(self):
        return [zip_format]


class TarGzConverter(Converter):

    requirements = []

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(TarGzDir(), name=node_name,
                            memory=12000)
        return convert_node, 'dirname', 'zipped'

    def input_formats(self):
        return [directory_format]

    def output_formats(self):
        return [targz_format]


class UnTarGzConverter(Converter):

    requirements = []

    def _get_convert_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(UnTarGzDir(), name=node_name,
                            memory=12000)
        return convert_node, 'gzipped', 'gunzipped'

    def input_formats(self):
        return [targz_format]

    def output_formats(self):
        return [directory_format]


# List all possible converters in order of preference
all_converters = [Dcm2niixConverter(), MrtrixConverter(), UnzipConverter(),
                  ZipConverter(), UnTarGzConverter(), TarGzConverter()]

# A dictionary to access all the formats by name
data_formats = dict(
    (f.name, f) for f in copy(globals()).itervalues()
    if isinstance(f, DataFormat))


data_formats_by_ext = dict(
    (f.extension, f) for f in data_formats.itervalues()
    if f.extension is not None)


data_formats_by_within_exts = dict(
    (f.within_dir_exts, f) for f in data_formats.itervalues()
    if f.within_dir_exts is not None)


def get_converter_node(dataset, dataset_name, output_format, source, workflow,
                       node_name, converters=None):
    if converters is None:
        converters = all_converters
    for converter in converters:
        if (dataset.format in converter.input_formats() and
            output_format in converter.output_formats() and
                converter.is_available):
            return converter.convert(workflow, source, dataset, dataset_name,
                                     node_name, output_format)
    raise NiAnalysisUsageError(
        "No available converters to convert between '{}' and '{}' formats."
        .format(dataset.format.name, output_format.name))