from base import MRIStudy
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (nifti_gz_format, text_matrix_format,
                                     text_format, directory_format)
from nianalysis.citations import fsl_cite
from nipype.interfaces import fsl
from nianalysis.requirements import fsl5_req
from nianalysis.study.base import set_dataset_specs
from .coregistered import CoregisteredStudy
from ..combined import CombinedStudy
from itertools import chain


class EPIStudy(MRIStudy):

    def brain_mask_pipeline(self, robust=True, threshold=0.2, **kwargs):
        super(EPIStudy, self).brain_mask_pipeline(
            robust=robust, threshold=threshold, **kwargs)

    def motion_alignment_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='MCFLIRT_pipeline',
            inputs=[DatasetSpec('epi', nifti_gz_format)],
            outputs=[DatasetSpec('epi_mc', nifti_gz_format),
                     DatasetSpec('epi_mc_mat', text_matrix_format),
                     DatasetSpec('epi_mc_par', text_format)],
            description=("Intra-epi volumes alignment."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        mcflirt = pipeline.create_node(fsl.MCFLIRT(), name='mcflirt',
                                       requirements=[fsl5_req])
        mcflirt.inputs.ref_vol = 0
        mcflirt.inputs.save_mats = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.output_type = 'NIFTI_GZ'
        mcflirt.inputs.out_file = 'epi_mc.nii.gz'
        pipeline.connect_input('epi', mcflirt, 'in_file')
        pipeline.connect_output('epi_mc', mcflirt, 'out_file')
        pipeline.connect_output('epi_mc_mat', mcflirt, 'mat_file')
        pipeline.connect_output('epi_mc_par', mcflirt, 'par_file')

        pipeline.assert_connected()
        return pipeline


    _dataset_specs = set_dataset_specs(
        DatasetSpec('epireg_mat', text_matrix_format, epireg_pipeline),
        DatasetSpec('epi_mc', nifti_gz_format, motion_alignment_pipeline),
        DatasetSpec('epi_mc_mat', directory_format, motion_alignment_pipeline),
        DatasetSpec('epi_mc_par', text_format, motion_alignment_pipeline),
        inherit_from=MRIStudy.dataset_specs())


class CoregisteredEPIStudy(CombinedStudy):

    sub_study_specs = {
        'epi': (EPIStudy, {
            'epi': 'primary',
            'epi_preproc': 'preproc',
            'epi_brain': 'masked'}),
        'reference': (MRIStudy, {
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_wmseg': 'wm_seg'}),
        'coreg': (CoregisteredStudy, {
            'epi_preproc': 'to_register',
            'ref_preproc': 'reference',
            'epi_qformed': 'qformed'})}

    epi_basic_preproc_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.basic_preproc_pipeline)

    def ref_basic_preproc_pipeline(self, **options):
        pipeline = self.TranslatedPipeline(
            'reference', self.reference.basic_preproc_pipeline(**options),
            self, default_options={'resolution': (1.0, 1.0, 1.0)})
        return pipeline

    epi_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    def epireg_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='EPIREG_pipeline',
            inputs=[DatasetSpec('masked', nifti_gz_format),
                    DatasetSpec('ref_brain', nifti_gz_format),
                    DatasetSpec('ref_head', nifti_gz_format),
                    DatasetSpec('ref_wmseg', nifti_gz_format)],
            outputs=[DatasetSpec('epi_epireg', nifti_gz_format),
                     DatasetSpec('epi_epireg_mat', text_matrix_format)],
            description=("Intra-subjects epi registration improved "
                         "using white matter boundaries."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        epireg = pipeline.create_node(fsl.epi.EpiReg(), name='epireg',
                                      requirements=[fsl5_req])

        epireg.inputs.out_base = 'epireg2ref'
        pipeline.connect_input('masked', epireg, 'epi')
        pipeline.connect_input('t1_brain', epireg, 'ref_brain')
        pipeline.connect_input('t1_head', epireg, 'ref_head')
        pipeline.connect_input('ref_wmseg', epireg, 'wmseg')

        pipeline.connect_output('epi_epireg', epireg, 'out_vol')
        pipeline.connect_output('epi_epireg_mat', epireg, 'epi2str_mat')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('epi', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('epi_preproc', nifti_gz_format,
                    epi_basic_preproc_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('epi_qformed', nifti_gz_format,
                    epi_qform_transform_pipeline),
        DatasetSpec('ref_head', nifti_gz_format),
        DatasetSpec('ref_wmseg', nifti_gz_format),
        DatasetSpec('epi_epireg', nifti_gz_format, epireg_pipeline),
        DatasetSpec('epi_epireg_mat', text_matrix_format,
                    epireg_pipeline))
