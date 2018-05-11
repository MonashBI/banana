from base import MRIStudy
from nipype.interfaces.fsl import TOPUP, ApplyTOPUP
from nianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, GenTopupConfigFiles)
from arcana.dataset import DatasetSpec, FieldSpec
from nianalysis.data_format import (nifti_gz_format, text_matrix_format,
                                    directory_format, par_format)
from nianalysis.citation import fsl_cite
from nipype.interfaces import fsl
from nianalysis.requirement import fsl509_req
from arcana.study.base import StudyMetaClass
from nianalysis.interfaces.custom.motion_correction import (
    MergeListMotionMat)
from arcana.option import OptionSpec
from nipype.interfaces.utility import Merge as merge_lists
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import BET, FUGUE


class EPIStudy(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('coreg_ref_preproc', nifti_gz_format),
        DatasetSpec('coreg_ref_wmseg', nifti_gz_format),
        DatasetSpec('reverse_phase', nifti_gz_format, optional=True),
        DatasetSpec('field_map_mag', nifti_gz_format, optional=True),
        DatasetSpec('field_map_phase', nifti_gz_format, optional=True),
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

    def basic_preproc_pipeline(self, **kwargs):

        if ('field_map_phase' in self.input_names and
                'field_map_mag' in self.input_names):
            return self._fugue_pipeline(**kwargs)
        elif 'reverse_phase' in self.input_names:
            return self._topup_pipeline(**kwargs)
        else:
            return super(EPIStudy, self).basic_preproc_pipeline(**kwargs)

    def _topup_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='basic_preproc_pipeline',
            inputs=[DatasetSpec('primary', nifti_gz_format),
                    DatasetSpec('reverse_phase', nifti_gz_format),
                    FieldSpec('ped', str),
                    FieldSpec('pe_angle', str)],
            outputs=[DatasetSpec('preproc', nifti_gz_format)],
            desc=("Topup distortion correction pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_in',
            requirements=[fsl509_req])
        pipeline.connect_input('primary', reorient_epi_in, 'in_file')

        reorient_epi_opposite = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_opposite',
            requirements=[fsl509_req])
        pipeline.connect_input('reverse_phase', reorient_epi_opposite,
                               'in_file')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        prep_dwi.inputs.topup = True
        pipeline.connect_input('ped', prep_dwi, 'pe_dir')
        pipeline.connect_input('pe_angle', prep_dwi, 'phase_offset')
        pipeline.connect(reorient_epi_in, 'out_file', prep_dwi, 'dwi')
        pipeline.connect(reorient_epi_opposite, 'out_file', prep_dwi,
                         'dwi1')
        ped1 = pipeline.create_node(GenTopupConfigFiles(),
                                    name='gen_config1')
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
        pipeline.connect(reorient_epi_in, 'out_file', in_apply_tp1, 'in1')
        apply_topup = pipeline.create_node(
            ApplyTOPUP(), name='applytopup1', requirements=[fsl509_req])
        apply_topup.inputs.method = 'jac'
        apply_topup.inputs.in_index = [1]
        pipeline.connect(in_apply_tp1, 'out', apply_topup, 'in_files')
        pipeline.connect(
            ped1, 'apply_topup_config', apply_topup, 'encoding_file')
        pipeline.connect(topup1, 'out_movpar', apply_topup,
                         'in_topup_movpar')
        pipeline.connect(
            topup1, 'out_fieldcoef', apply_topup, 'in_topup_fieldcoef')

        pipeline.connect_output('preproc', apply_topup, 'out_corrected')
        return pipeline

    def _fugue_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='basic_preproc_pipeline',
            inputs=[DatasetSpec('primary', nifti_gz_format),
                    DatasetSpec('field_map_mag', nifti_gz_format),
                    DatasetSpec('field_map_phase', nifti_gz_format)],
            outputs=[DatasetSpec('preproc', nifti_gz_format)],
            desc=("Fugue distortion correction pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_in',
            requirements=[fsl509_req])
        pipeline.connect_input('primary', reorient_epi_in, 'in_file')
        fm_mag_reorient = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_fm_mag',
            requirements=[fsl509_req])
        pipeline.connect_input('field_map_mag', fm_mag_reorient, 'in_file')
        fm_phase_reorient = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_fm_phase',
            requirements=[fsl509_req])
        pipeline.connect_input('field_map_phase', fm_phase_reorient,
                               'in_file')
        bet = pipeline.create_node(BET(), name="bet", wall_time=5,
                                   requirements=[fsl509_req])
        bet.inputs.robust = True
        pipeline.connect(fm_mag_reorient, 'out_file', bet, 'in_file')
        create_fmap = pipeline.create_node(
            PrepareFieldmap(), name="prepfmap", wall_time=5,
            requirements=[fsl509_req])
        create_fmap.inputs.delta_TE = 2.46
        pipeline.connect(bet, "out_file", create_fmap, "in_magnitude")
        pipeline.connect(fm_phase_reorient, 'out_file', create_fmap,
                         'in_phase')

        fugue = pipeline.create_node(FUGUE(), name='fugue', wall_time=5,
                                     requirements=[fsl509_req])
        fugue.inputs.unwarp_direction = 'x'
        fugue.inputs.dwell_time = 0.000275
        fugue.inputs.unwarped_file = 'example_func.nii.gz'
        pipeline.connect(create_fmap, 'out_fieldmap', fugue,
                         'fmap_in_file')
        pipeline.connect(reorient_epi_in, 'out_file', fugue, 'in_file')
        pipeline.connect_output('preproc', fugue, 'unwarped_file')
        return pipeline
