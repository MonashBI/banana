from nipype.interfaces.fsl.model import FEAT, MELODIC
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import BET, FUGUE, FLIRT
from nipype.interfaces.afni.preprocess import Volreg
from nipype.interfaces.fsl.utils import SwapDimensions
from nianalysis.interfaces.fsl import MelodicL1FSF, FSLFIX
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import set_dataset_specs
from ..base import MRIStudy
from nianalysis.requirements import fsl5_req, fix_req, ants2_req, afni_req
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import (
    nifti_gz_format, rdata_format, directory_format,
    zip_format)
from nianalysis.interfaces.utils import OptiBET, AntsRegSyn, SetANTsPath
from nianalysis.interfaces.afni import Tproject
import os
import subprocess as sp
from logging import raiseExceptions


class FunctionalMRIStudy(MRIStudy):

    def feat_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='feat',
            inputs=[DatasetSpec('field_map_mag', nifti_gz_format),
                    DatasetSpec('field_map_phase', nifti_gz_format),
                    DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('rs_fmri', nifti_gz_format),
                    DatasetSpec('rs_fmri_ref', nifti_gz_format)],
            outputs=[DatasetSpec('feat_dir', directory_format)],
            description="MELODIC Level 1",
            default_options={'brain_thresh_percent': 5},
            version=1,
            citations=[fsl_cite],
            options=options)
        swap_dims = pipeline.create_node(SwapDimensions(), "swap_dims")
        swap_dims.inputs.new_dims = ('LR', 'PA', 'IS')
        pipeline.connect_input('t1', swap_dims, 'in_file')

        bet = pipeline.create_node(interface=BET(), name="bet",
                                   requirements=[fsl5_req])
        bet.inputs.frac = 0.2
        bet.inputs.reduce_bias = True
        pipeline.connect_input('field_map_mag', bet, 'in_file')

        bet2 = pipeline.create_node(BET(), "bet2", [fsl5_req])
        bet2.inputs.frac = 0.2
        bet2.inputs.reduce_bias = True
        bet2.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(swap_dims, "out_file", bet2, "in_file")
        create_fmap = pipeline.create_node(PrepareFieldmap(), "prepfmap")
#       create_fmap.inputs.in_magnitude = fmap_mag[0]

        create_fmap.inputs.delta_TE = 2.46
        pipeline.connect(bet, "out_file", create_fmap, "in_magnitude")
        pipeline.connect_input('field_map_phase', create_fmap, 'in_phase')

        mel = MelodicL1FSF()
        mel.inputs.brain_thresh = pipeline.option('brain_thresh_percent')
        ml1 = pipeline.create_node(mel, "mL1FSF", [fsl5_req])
        ml1.inputs.tr = 0.754
        ml1.inputs.dwell_time = 0.39
        ml1.inputs.te = 21
        ml1.inputs.unwarp_dir = "x"
        ml1.inputs.sfwhm = 3
        ml1.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect_input('rs_fmri', ml1, 'fmri')
        pipeline.connect_input('rs_fmri_ref', ml1, 'fmri_ref')
#        ml1.inputs.fmap_mag = [0]
#        ml1.inputs.structural = struct[0]
        ml1.inputs.high_pass = 75
        pipeline.connect(create_fmap, "out_fieldmap", ml1, "fmap")
        pipeline.connect(bet, "out_file", ml1, "fmap_mag")
        pipeline.connect(bet2, "out_file", ml1, "structural")
        ml1.inputs.output_dir = ("/mnt/rar/project/test_ASPREE/test_pipeline"
                                 "/T1/melodic.ica")
        # fix next
        feat = pipeline.create_node(FEAT(), "featL1", [fsl5_req])
        feat.inputs.terminal_output = 'none'
        feat.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(ml1, 'fsf_file', feat, 'fsf_file')
        pipeline.connect_output('feat_dir', feat, 'feat_dir')

        pipeline.assert_connected()
        return pipeline

    def fix_pipeline(self, **options):

        pipeline = self._create_pipeline(
            name='fix',
            # inputs=['fear_dir', 'train_data'],
            inputs=[DatasetSpec('feat_dir', directory_format),
                    DatasetSpec('train_data', rdata_format)],
            outputs=[DatasetSpec('cleaned_file', nifti_gz_format)],
            description=("Automatic classification and removal of noisy"
                         "components from the rsfMRI data"),
            default_options={'component_threshold': 20, 'motion_reg': True},
            version=1,
            requirements=[fsl5_req, fix_req],
            citations=[fsl_cite],
            approx_runtime=10,
            options=options)
        finter = FSLFIX()

        fix = pe.Node(interface=finter, name="fix")
        pipeline.connect_input("feat_dir", fix, "feat_dir")
        pipeline.connect_input("train_data", fix, "train_data")
        finter.inputs.component_threshold = pipeline.option(
            'component_threshold')
        finter.inputs.motion_reg = pipeline.option('motion_reg')

        pipeline.connect_output('cleaned_file', fix, 'output')

        pipeline.assert_connected()
        return pipeline

    def ASPREE_preproc(self, **options):

        pipeline = self._create_pipeline(
            name='ASPREE_preprocessing',
            inputs=[DatasetSpec('field_map_mag', nifti_gz_format),
                    DatasetSpec('field_map_phase', nifti_gz_format),
                    DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('rs_fmri', nifti_gz_format),
                    DatasetSpec('rs_fmri_ref', nifti_gz_format)],
            outputs=[DatasetSpec('filtered_data', nifti_gz_format)],
            description=("Funtional MRI preprocessing using state-of-the-art"
                         "toolboxes"),
            default_options={'brain_thresh_percent': 5,
                             'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            requirements=[fsl5_req, ants2_req, afni_req],
            citations=[fsl_cite],
            approx_runtime=60,
            options=options)
        try:
            cmd = 'which ANTS'
            antspath = sp.check_output(cmd, shell=True)
            antspath = '/'.join(antspath.split('/')[0:-2])
            os.environ['ANTSPATH'] = antspath
        except ImportError:
            print "NO ANTs module found. Please ensure to have it in you PATH."
        swap_dims = pe.Node(interface=SwapDimensions(), name="swap_dims")
        swap_dims.inputs.new_dims = ('LR', 'PA', 'IS')
        pipeline.connect_input('t1', swap_dims, 'in_file')

        bet = pe.Node(interface=BET(), name="bet")
        bet.inputs.robust = True
        pipeline.connect_input('field_map_mag', bet, 'in_file')

        bet_rsfmri = pe.Node(interface=BET(), name="bet_rsfmri")
        bet_rsfmri.inputs.robust = True
        bet_rsfmri.inputs.frac = 0.4
        bet_rsfmri.inputs.mask = True
        pipeline.connect_input('rsfmri', bet_rsfmri, 'in_file')

        create_fmap = pe.Node(interface=PrepareFieldmap(), name="prepfmap")
        create_fmap.inputs.delta_TE = 2.46
        pipeline.connect(bet, "out_file", create_fmap, "in_magnitude")
        pipeline.connect_input('field_map_phase', create_fmap, 'in_phase')

        fugue = pe.Node(interface=FUGUE(), name='fugue')
        fugue.inputs.unwarp_direction = 'x'
        fugue.inputs.dwell_time = 0.00039
        fugue.inputs.unwarped_file = 'unwarped_rsfmri'
        pipeline.connect_input(create_fmap, 'out_fieldmap', fugue,
                               'fmap_in_file')
        pipeline.connect_input('rs_fmri', fugue, 'in_file')

        optibet = pe.Node(interface=OptiBET(), name='optiBET')
        pipeline.connect_input('t1', optibet, 'input_file')

        antreg_rsfmri = pe.Node(interface=AntsRegSyn, name='ANTsReg')
        antreg_rsfmri.inputs.num_dimensions = 3
        antreg_rsfmri.inputs.transformation = 'r'
        antreg_rsfmri.inputs.out_prefix = 'epi2T1'
        pipeline.connect_input(
            optibet, 'betted_file', antreg_rsfmri, 'ref_file')
        pipeline.connect_input(
            bet_rsfmri, 'out_file', antreg_rsfmri, 'input_file')

        antreg_t1 = pe.Node(interface=AntsRegSyn, name='ANTsReg_T1')
        antreg_t1.inputs.num_dimensions = 3
        antreg_t1.inputs.transformation = 's'
        antreg_t1.inputs.out_prefix = 'T12MNI'
        pipeline.connect_input(
            optibet, 'betted_file', antreg_t1, 'input_file')
        pipeline.connect_input(
            pipeline.option('MNI_template'), antreg_t1, 'ref_file')

        flirt_t1 = pe.Node(interface=FLIRT(), name='FLIRT_T1')
        flirt_t1.inputs.dof = 6
        flirt_t1.inputs.out_matrix_file = 'example2hires.mat'
        pipeline.connect_input('rs_fmri', flirt_t1, 'in_file')
        pipeline.connect_input(pipeline.option('MNI_template'), flirt_t1,
                               'reference')
        afni_mc = pe.Node(interface=Volreg(), name='AFNI_MC')
        afni_mc.inputs.zpad = 1
        afni_mc.inputs.out_file = 'rsfmri_mc.nii.gz'
        afni_mc.inputs.oned_file = 'prefiltered_func_data_mcf.par'
        afni_mc.inputs.oned_matrix_save = 'motion_matrices.mat'
        pipeline.connect_input(fugue, 'unwarped_file', afni_mc, 'in_file')

        filt = pe.Node(interface=Tproject, name='filtering')
        filt.inputs.stopband = (0, 0.01)
        filt.inputs.delta_t = 0.754
        filt.inputs.polort = 3
        filt.inputs.blur = 3
        pipeline.connect_input(afni_mc, 'out_file', filt, 'in_file')
        pipeline.connect_input(bet_rsfmri, 'mask_file', filt, 'mask')

        melodicL1 = pe.Node(interface=MELODIC(), name='MelodicL1')

    _dataset_specs = set_dataset_specs(
        DatasetSpec('field_map_mag', nifti_gz_format),
        DatasetSpec('field_map_phase', nifti_gz_format),
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('rs_fmri', nifti_gz_format),
        DatasetSpec('rs_fmri_ref', nifti_gz_format),
        DatasetSpec('feat_dir', zip_format, feat_pipeline),
        DatasetSpec('train_data', rdata_format),
        DatasetSpec('cleaned_file', nifti_gz_format, fix_pipeline))
