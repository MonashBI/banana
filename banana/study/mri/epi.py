from .base import MriStudy
from nipype.interfaces.fsl import TOPUP, ApplyTOPUP
from banana.interfaces.custom.motion_correction import (
    PrepareDWI, GenTopupConfigFiles)
from arcana.data import AcquiredFilesetSpec, FilesetSpec, FieldSpec
from banana.file_format import (
    nifti_gz_format, text_matrix_format, directory_format,
    par_format, motion_mats_format)
from banana.citation import fsl_cite
from nipype.interfaces import fsl
from banana.requirement import fsl509_req
from arcana.study.base import StudyMetaClass
from banana.interfaces.custom.motion_correction import (
    MergeListMotionMat, MotionMatCalculation)
from arcana.parameter import ParameterSpec
from nipype.interfaces.utility import Merge as merge_lists
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import BET, FUGUE


from banana.file_format import STD_IMAGE_FORMATS


class EpiStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        AcquiredFilesetSpec('coreg_ref_preproc', STD_IMAGE_FORMATS,
                            optional=True),
        AcquiredFilesetSpec('coreg_ref_wmseg', STD_IMAGE_FORMATS,
                            optional=True),
        AcquiredFilesetSpec('reverse_phase', STD_IMAGE_FORMATS, optional=True),
        AcquiredFilesetSpec('field_map_mag', STD_IMAGE_FORMATS,
                            optional=True),
        AcquiredFilesetSpec('field_map_phase', STD_IMAGE_FORMATS,
                            optional=True),
        FilesetSpec('moco', nifti_gz_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('align_mats', directory_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('moco_par', par_format,
                    'intrascan_alignment_pipeline'),
        FieldSpec('field_map_delta_te', float,
                  'field_map_time_info_pipeline')]

    add_param_specs = [
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.2),
        ParameterSpec('bet_reduce_bias', False),
        ParameterSpec('fugue_echo_spacing', 0.000275),
        ParameterSpec('linear_reg_method', 'epireg')]

    def linear_coregistration_pipeline(self, **kwargs):
        if self.branch('linear_reg_method', 'epireg'):
            return self._epireg_linear_coregistration_pipeline(**kwargs)
        else:
            return super(EpiStudy, self).linear_coregistration_pipeline(
                **kwargs)

    def _epireg_linear_coregistration_pipeline(self, **kwargs):

            # inputs=[FilesetSpec('brain', nifti_gz_format),
            #         FilesetSpec('coreg_ref_brain', nifti_gz_format),
            #         FilesetSpec('coreg_ref_preproc', nifti_gz_format),
            #         FilesetSpec('coreg_ref_wmseg', nifti_gz_format)],
            # outputs=[FilesetSpec('coreg_brain', nifti_gz_format),
            #          FilesetSpec('coreg_matrix', text_matrix_format)],

        pipeline = self.pipeline(
            name='linear_coreg',
            desc=("Intra-subjects epi registration improved using white "
                  "matter boundaries."),
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

    def intrascan_alignment_pipeline(self, **kwargs):

         # inputs=[FilesetSpec('preproc', nifti_gz_format)],
         #   outputs=[FilesetSpec('moco', nifti_gz_format),
         #            FilesetSpec('align_mats', directory_format),
         #            FilesetSpec('moco_par', par_format)],

        pipeline = self.pipeline(
            name='MCFLIRT_pipeline',
            desc=("Intra-epi volumes alignment."),
            citations=[fsl_cite],
            **kwargs)
        mcflirt = pipeline.add('mcflirt', fsl.MCFLIRT(),
                               requirements=[fsl509_req])
        mcflirt.inputs.ref_vol = 0
        mcflirt.inputs.save_mats = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.output_type = 'NIFTI_GZ'
        mcflirt.inputs.out_file = 'moco.nii.gz'
        pipeline.connect_input('preproc', mcflirt, 'in_file')
        pipeline.connect_output('moco', mcflirt, 'out_file')
        pipeline.connect_output('moco_par', mcflirt, 'par_file')

        merge = pipeline.add('merge', MergeListMotionMat())
        pipeline.connect(mcflirt, 'mat_file', merge, 'file_list')
        pipeline.connect_output('align_mats', merge, 'out_dir')

        return pipeline

    def field_map_time_info_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='field_map_time_info_pipeline',
            inputs=[DatasetSpec('field_map_mag', dicom_format)],
            outputs=[FieldSpec('field_map_delta_te', float)],
            desc=("Pipeline to extract delta TE from field map "
                  "images, if provided"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        delta_te = pipeline.create_node(FieldMapTimeInfo(),
                                        name='extract_delta_te')
        pipeline.connect_input('field_map_mag', delta_te, 'fm_mag')
        pipeline.connect_output('field_map_delta_te', delta_te, 'delta_te')

        return pipeline

    def preprocess_pipeline(self, **kwargs):

        if ('field_map_phase' in self.input_names and
                'field_map_mag' in self.input_names):
            return self._fugue_pipeline(**kwargs)
        elif 'reverse_phase' in self.input_names:
            return self._topup_pipeline(**kwargs)
        else:
            return super(EpiStudy, self).preprocess_pipeline(**kwargs)

    def _topup_pipeline(self, **kwargs):

#            inputs=[FilesetSpec('magnitude', nifti_gz_format),
#                    FilesetSpec('reverse_phase', nifti_gz_format),
#                    FieldSpec('ped', str),
#                    FieldSpec('pe_angle', str)],
#            outputs=[FilesetSpec('preproc', nifti_gz_format)],


        pipeline = self.pipeline(
            name='preprocess_pipeline',
            desc=("Topup distortion correction pipeline"),
            citations=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_in',
            requirements=[fsl509_req])
        pipeline.connect_input('magnitude', reorient_epi_in, 'in_file')

        reorient_epi_opposite = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_opposite',
            requirements=[fsl509_req])
        pipeline.connect_input('reverse_phase', reorient_epi_opposite,
                               'in_file')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        prep_dwi.inputs.topup = True
        pipeline.connect_input('ped', prep_dwi, 'pe_dir')
        pipeline.connect_input('pe_angle', prep_dwi, 'ped_polarity')
        pipeline.connect(reorient_epi_in, 'out_file', prep_dwi, 'dwi')
        pipeline.connect(reorient_epi_opposite, 'out_file', prep_dwi,
                         'dwi1')
        ped = pipeline.create_node(GenTopupConfigFiles(), name='gen_config')
        pipeline.connect(prep_dwi, 'pe', ped, 'ped')
        merge_outputs = pipeline.create_node(merge_lists(2),
                                             name='merge_files')
        pipeline.connect(prep_dwi, 'main', merge_outputs, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs, 'in2')
        merge = pipeline.create_node(fsl_merge(), name='fsl_merge',
                                     requirements=[fsl509_req])
        merge.inputs.dimension = 't'
        pipeline.connect(merge_outputs, 'out', merge, 'in_files')
        topup = pipeline.create_node(TOPUP(), name='topup',
                                     requirements=[fsl509_req])
        pipeline.connect(merge, 'merged_file', topup, 'in_file')
        pipeline.connect(ped, 'config_file', topup, 'encoding_file')
        in_apply_tp = pipeline.create_node(merge_lists(1),
                                           name='in_apply_tp')
        pipeline.connect(reorient_epi_in, 'out_file', in_apply_tp, 'in1')
        apply_topup = pipeline.create_node(
            ApplyTOPUP(), name='applytopup', requirements=[fsl509_req])
        apply_topup.inputs.method = 'jac'
        apply_topup.inputs.in_index = [1]
        pipeline.connect(in_apply_tp, 'out', apply_topup, 'in_files')
        pipeline.connect(
            ped, 'apply_topup_config', apply_topup, 'encoding_file')
        pipeline.connect(topup, 'out_movpar', apply_topup,
                         'in_topup_movpar')
        pipeline.connect(
            topup, 'out_fieldcoef', apply_topup, 'in_topup_fieldcoef')

        pipeline.connect_output('preproc', apply_topup, 'out_corrected')
        return pipeline

    def _fugue_pipeline(self, **kwargs):

#            inputs=[FilesetSpec('magnitude', nifti_gz_format),
#                    FilesetSpec('field_map_mag', nifti_gz_format),
#                    FilesetSpec('field_map_phase', nifti_gz_format)],
#            outputs=[FilesetSpec('preproc', nifti_gz_format)],


        pipeline = self.pipeline(
            name='preprocess_pipeline',
            desc=("Fugue distortion correction pipeline"),
            references=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='reorient_epi_in',
            requirements=[fsl509_req])
        pipeline.connect_input('magnitude', reorient_epi_in, 'in_file')
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
#         create_fmap.inputs.delta_TE = 2.46
        pipeline.connect_input('field_map_delta_te', create_fmap,
                               'delta_TE')
        pipeline.connect(bet, "out_file", create_fmap, "in_magnitude")
        pipeline.connect(fm_phase_reorient, 'out_file', create_fmap,
                         'in_phase')

        fugue = pipeline.create_node(FUGUE(), name='fugue', wall_time=5,
                                     requirements=[fsl509_req])
        fugue.inputs.unwarp_direction = 'x'
        fugue.inputs.dwell_time = self.parameter('fugue_echo_spacing')
        fugue.inputs.unwarped_file = 'example_func.nii.gz'
        pipeline.connect(create_fmap, 'out_fieldmap', fugue,
                         'fmap_in_file')
        pipeline.connect(reorient_epi_in, 'out_file', fugue, 'in_file')
        pipeline.connect_output('preproc', fugue, 'unwarped_file')
        return pipeline

    def motion_mat_pipeline(self, **kwargs):

#        inputs = [FilesetSpec('coreg_matrix', text_matrix_format),
#                  FilesetSpec('qform_mat', text_matrix_format)]
#            inputs=inputs,
#            outputs=[FilesetSpec('motion_mats', motion_mats_format)],
        
#        if 'reverse_phase' not in self.input_names:
#            inputs.append(FilesetSpec('align_mats', directory_format))
        pipeline = self.pipeline(
            name='motion_mat_calculation',
            desc=("Motion matrices calculation"),
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('coreg_matrix', mm, 'reg_mat')
        pipeline.connect_input('qform_mat', mm, 'qform_mat')
        if 'reverse_phase' not in self.input_names:
            pipeline.connect_input('align_mats', mm, 'align_mats')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline
