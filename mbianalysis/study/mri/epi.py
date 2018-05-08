from base import MRIStudy
from nipype.interfaces.fsl import (ExtractROI, TOPUP, ApplyTOPUP)
from mbianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, CheckDwiNames, GenTopupConfigFiles)
from nianalysis.dataset import DatasetSpec, FieldSpec
from mbianalysis.data_format import (nifti_gz_format, text_matrix_format,
                                     directory_format, par_format,
                                     motion_mats_format)
from mbianalysis.citation import fsl_cite
from nipype.interfaces import fsl
from mbianalysis.requirement import fsl509_req
from nianalysis.study.base import StudyMetaClass
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, MergeListMotionMat)
from nianalysis.option import OptionSpec
from nipype.interfaces.utility import Merge as merge_lists
from nipype.interfaces.fsl.utils import Merge as fsl_merge


class EPIStudy(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('coreg_ref_preproc', nifti_gz_format),
        DatasetSpec('coreg_ref_wmseg', nifti_gz_format),
        DatasetSpec('moco', nifti_gz_format,
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
            inputs=[DatasetSpec('brain', nifti_gz_format),
                    DatasetSpec('coreg_ref_brain', nifti_gz_format),
                    DatasetSpec('coreg_ref_preproc', nifti_gz_format),
                    DatasetSpec('coreg_ref_wmseg', nifti_gz_format)],
            outputs=[DatasetSpec('coreg_brain', nifti_gz_format),
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
            inputs=[DatasetSpec('preproc', nifti_gz_format)],
            outputs=[DatasetSpec('moco', nifti_gz_format),
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
        return (super(EPIStudy, self).motion_mat_pipeline_factory(
            align_mats='moco_mat', **kwargs))
    
    def basic_preproc_pipeline(self, in_file_name, method='topup', **kwargs):

        pipeline = self.create_pipeline(
            name='basic_preproc_pipeline',
            inputs=[DatasetSpec(in_file_name, nifti_gz_format)],
            outputs=[DatasetSpec('preproc', nifti_gz_format)],
            desc=("Dimensions swapping to ensure that all the images "
                  "have the same orientations."),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        swap = pipeline.create_node(fsl.utils.Reorient2Std(),
                                    name='fslreorient2std',
                                    requirements=[fsl509_req])
        pipeline.connect_input(in_file_name, swap, 'in_file')
        
    def _distortion_correction_pipeline(
            self, epi_in, epi_opposite_ped, epi_in_pe_dir, epi_in_pe_angle,
            out_name, **kwargs):

        pipeline = self.create_pipeline(
            name='topup_preproc',
            inputs=[DatasetSpec(epi_in, nifti_gz_format),
                    DatasetSpec(epi_opposite_ped, nifti_gz_format),
                    FieldSpec(epi_in_pe_dir, dtype=str),
                    FieldSpec(epi_in_pe_angle, dtype=str)],
            outputs=[DatasetSpec(out_name, nifti_gz_format)],
            desc=("Topup distortion correction pipeline."),
            version=1,
            citations=[],
            **kwargs)

        reorient1 = pipeline.create_node(fsl.utils.Reorient2Std(),
                                    name='fslreorient2std',
                                    requirements=[fsl509_req])
        pipeline.connect_input(epi_in, reorient1, 'in_file')
        reorient2 = pipeline.create_node(fsl.utils.Reorient2Std(),
                                    name='fslreorient2std',
                                    requirements=[fsl509_req])
        pipeline.connect_input(epi_opposite_ped, reorient2, 'in_file')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        prep_dwi.inputs.topup = True
        pipeline.connect_input(epi_in_pe_dir, prep_dwi, 'pe_dir')
        pipeline.connect_input(epi_in_pe_angle, prep_dwi, 'phase_offset')
        pipeline.connect_input(epi_in, prep_dwi, 'dwi')
        pipeline.connect_input(epi_opposite_ped, prep_dwi, 'dwi1')
        ped1 = pipeline.create_node(GenTopupConfigFiles(), name='gen_config1')
        pipeline.connect(prep_dwi, 'pe', ped1, 'ped')
        merge_outputs1 = pipeline.create_node(merge_lists(2),
                                              name='merge_files1')
        pipeline.connect(prep_dwi, 'main', merge_outputs1, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs1, 'in2')
        merge1 = pipeline.create_node(fsl_merge(), name='fsl_merge1',
                                      requirements=[fsl509_req])
        merge1.inputs.dimension = 't'
        pipeline.connect(merge_outputs1, 'out', merge1, 'in_files')
        topup1 = pipeline.create_node(TOPUP(), name='topup1',
                                      requirements=[fsl509_req])
        pipeline.connect(merge1, 'merged_file', topup1, 'in_file')
        pipeline.connect(ped1, 'config_file', topup1, 'encoding_file')
        in_apply_tp1 = pipeline.create_node(merge_lists(1),
                                            name='in_apply_tp1')
        pipeline.connect_input(epi_in, in_apply_tp1, 'in1')
        apply_topup1 = pipeline.create_node(ApplyTOPUP(), name='applytopup1',
                                            requirements=[fsl509_req])
        apply_topup1.inputs.method = 'jac'
        apply_topup1.inputs.in_index = [1]
        pipeline.connect(in_apply_tp1, 'out', apply_topup1, 'in_files')
        pipeline.connect(
            ped1, 'apply_topup_config', apply_topup1, 'encoding_file')
        pipeline.connect(topup1, 'out_movpar', apply_topup1, 'in_topup_movpar')
        pipeline.connect(
            topup1, 'out_fieldcoef', apply_topup1, 'in_topup_fieldcoef')

        pipeline.connect_output(out_name, apply_topup1, 'out_corrected')
        return pipeline
