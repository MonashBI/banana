import os.path as op
from arcana.study.base import Study, StudyMetaClass
from arcana import ParameterSpec, DatasetSpec
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
        DatasetSpec('ct_reg_dicom', dicom_format, 'nifti2dcm_conversion_pipeline')]

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

        list_niftis = pipeline.create_node(ListDir(), name='list_niftis')
        reorient_niftis = pipeline.create_node(
            ReorientUmap(), name='reorient_niftis', requirements=[mrtrix3_req])

        nii2dicom = pipeline.create_map_node(
            Nii2Dicom(), name='nii2dicom',
            iterfield=['in_file'], wall_time=20)
#         nii2dicom.inputs.extension = 'Frame'
        list_dicoms = pipeline.create_node(ListDir(), name='list_dicoms')
        list_dicoms.inputs.sort_key = dicom_fname_sort_key
        copy2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        copy2dir.inputs.extension = 'Frame'
        # Connect nodes
        pipeline.connect(list_niftis, 'files', reorient_niftis, 'niftis')
        pipeline.connect(reorient_niftis, 'reoriented_umaps', nii2dicom,
                         'in_file')
        pipeline.connect(list_dicoms, 'files', nii2dicom, 'reference_dicom')
        pipeline.connect(nii2dicom, 'out_file', copy2dir, 'in_files')
        # Connect inputs
        pipeline.connect_input('dicom_ref', list_niftis, 'directory')
        pipeline.connect_input('ct_umap', list_dicoms, 'directory')
        pipeline.connect_input('ct_umap', reorient_niftis, 'umap')
        # Connect outputs
        pipeline.connect_output('ct_reg_dicom', copy2dir, 'out_dir')
        return pipeline
    