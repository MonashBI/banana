from base import MRIStudy
from nianalysis.dataset import DatasetSpec
from mbianalysis.data_format import (mrconvert_nifti_gz_format, text_matrix_format,
                                     text_format, directory_format, par_format,
                                     motion_mats_format)
from mbianalysis.citation import fsl_cite
from nipype.interfaces import fsl
from mbianalysis.requirement import fsl509_req
from nianalysis.study.base import StudyMetaClass
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, MergeListMotionMat)
from nianalysis.option import OptionSpec


class EPIStudy(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('coreg_ref_preproc', mrconvert_nifti_gz_format),
        DatasetSpec('coreg_ref_wmseg', mrconvert_nifti_gz_format),
        DatasetSpec('moco', mrconvert_nifti_gz_format,
                    'motion_alignment_pipeline'),
        DatasetSpec('moco_mat', directory_format,
                    'motion_alignment_pipeline'),
        DatasetSpec('moco_par', par_format,
                    'motion_alignment_pipeline')]

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.2),
        OptionSpec('bet_reduce_bias', False)]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(EPIStudy, self).
                header_info_extraction_pipeline_factory(
                    'primary', **kwargs))

    def linear_coregistration_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='EPIREG_pipeline',
            inputs=[DatasetSpec('brain', mrconvert_nifti_gz_format),
                    DatasetSpec('coreg_ref_brain', mrconvert_nifti_gz_format),
                    DatasetSpec('coreg_ref_preproc', mrconvert_nifti_gz_format),
                    DatasetSpec('coreg_ref_wmseg', mrconvert_nifti_gz_format)],
            outputs=[DatasetSpec('coreg_brain', mrconvert_nifti_gz_format),
                     DatasetSpec('coreg_matrix', text_matrix_format)],
            desc=("Intra-subjects epi registration improved using white "
                  "matter boundaries."),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        epireg = pipeline.create_node(fsl.epi.EpiReg(), name='epireg',
                                      requirements=[fsl509_req])

        epireg.inputs.out_base = 'epireg2ref'
        pipeline.connect_input('brain', epireg, 'epi')
        pipeline.connect_input('coreg_ref_brain', epireg, 't1_brain')
        pipeline.connect_input('coreg_ref_preproc', epireg, 't1_head')
        pipeline.connect_input('coreg_ref_wmseg', epireg, 'wmseg')

        pipeline.connect_output('coreg_brain', epireg, 'out_file')
        pipeline.connect_output('coreg_matrix', epireg, 'epi2str_mat')
        return pipeline

    def motion_alignment_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='MCFLIRT_pipeline',
            inputs=[DatasetSpec('preproc', mrconvert_nifti_gz_format)],
            outputs=[DatasetSpec('moco', mrconvert_nifti_gz_format),
                     DatasetSpec('moco_mat', directory_format),
                     DatasetSpec('moco_par', par_format)],
            desc=("Intra-epi volumes alignment."),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        mcflirt = pipeline.create_node(fsl.MCFLIRT(), name='mcflirt',
                                       requirements=[fsl509_req])
        mcflirt.inputs.ref_vol = 0
        mcflirt.inputs.save_mats = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.output_type = 'NIFTI_GZ'
        mcflirt.inputs.out_file = 'moco.nii.gz'
        pipeline.connect_input('preproc', mcflirt, 'in_file')
        pipeline.connect_output('moco', mcflirt, 'out_file')
        pipeline.connect_output('moco_par', mcflirt, 'par_file')

        merge = pipeline.create_node(MergeListMotionMat(), name='merge')
        pipeline.connect(mcflirt, 'mat_file', merge, 'file_list')
        pipeline.connect_output('moco_mat', merge, 'out_dir')

        return pipeline

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('coreg_matrix', text_matrix_format),
                    DatasetSpec('qform_mat', text_matrix_format),
                    DatasetSpec('moco_mat', directory_format)],
            outputs=[DatasetSpec('motion_mats', motion_mats_format)],
            desc=("EPI Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('coreg_matrix', mm, 'reg_mat')
        pipeline.connect_input('qform_mat', mm, 'qform_mat')
        pipeline.connect_input('moco_mat', mm, 'align_mats')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline
