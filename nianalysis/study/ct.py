import os.path as op
from arcana.study.base import Study, StudyMetaClass
from arcana import ParameterSpec, DatasetSpec
from nianalysis.interfaces.mrtrix import MRConvert
from nianalysis.file_format import dicom_format, nifti_gz_format
from nianalysis.interfaces.custom.motion_correction import ReorientUmap
from nianalysis.requirement import mrtrix3_req
from nianalysis.interfaces.converters import Nii2Dicom
from arcana.interfaces.utils import CopyToDir, ListDir, dicom_fname_sort_key


template_path = op.abspath(
    op.join(op.dirname(__file__).split('arcana')[0],
            'arcana', 'reference_data'))


class CtStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        DatasetSpec('ct', nifti_gz_format),
        DatasetSpec('dicom_ref', dicom_format),
        DatasetSpec('ct_umap', nifti_gz_format),
        DatasetSpec('ct_reg', nifti_gz_format, 'registration_pipeline'),
        DatasetSpec('ct_reg_dicom', dicom_format,
                    'nifti2dcm_conversion_pipeline')]

    add_parameter_specs = []

    def ct2umap_pipeline(self, **kwargs):
        pipeline = self.create_pipeline(
            name='ct2umap',
            inputs=[],
            outputs=[],
            desc=(),
            version=1,
            citations=(),
            **kwargs)
        return pipeline

    def registration_pipeline(self, **kwargs):
        pipeline = self.create_pipeline(
            name='registration',
            inputs=[],
            outputs=[],
            desc=(),
            version=1,
            citations=(),
            **kwargs)
        return pipeline

    def nifti2dcm_conversion_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='conversion_to_dicom',
            inputs=[DatasetSpec('ct_umap', nifti_gz_format),
                    DatasetSpec('dicom_ref', dicom_format)],
            outputs=[DatasetSpec('ct_reg_dicom', dicom_format)],
            desc=(
                "Convert aligned CT from nifti to dicom format - "
                "parallel implementation"),
            version=1,
            citations=(),
            **kwargs)

        # Restride nifti image so that it matches dicom reference
        mrconvert = pipeline.create_node(
            MRConvert(), name='reorient_nifti', requirements=[mrtrix3_req])
        # Node to copy nifti images to reference dicom
        nii2dicom = pipeline.create_node(
            Nii2Dicom(), name='nii2dicom', wall_time=20)
        # List DICOMs from input directory
        list_dicoms = pipeline.create_node(ListDir(), name='list_dicoms')
        list_dicoms.inputs.sort_key = dicom_fname_sort_key
        # Connect nodes
        pipeline.connect(mrconvert, 'out_file', nii2dicom, 'in_file')
        pipeline.connect(list_dicoms, 'files', nii2dicom, 'reference_dicom')
        # Connect inputs
        pipeline.connect_input('ct_umap', mrconvert, 'in_file')
        pipeline.connect_input('dicom_ref', mrconvert, 'stride')
        pipeline.connect_input('dicom_ref', list_dicoms, 'directory')
        # Connect outputs
        pipeline.connect_output('ct_reg_dicom', nii2dicom, 'out_file')
        return pipeline
