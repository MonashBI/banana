from nipype.interfaces.fsl.model import FEAT, MELODIC
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import (
    BET, FUGUE, FLIRT, FNIRT, ApplyWarp)
from nipype.interfaces.afni.preprocess import Volreg
from nipype.interfaces.fsl.utils import SwapDimensions, InvWarp, ImageMaths
from nianalysis.interfaces.fsl import MelodicL1FSF, FSLFIX

from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import set_dataset_specs
from ..base import MRIStudy
from nianalysis.requirements import fsl5_req, fix_req
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import (
    nifti_gz_format, rdata_format, directory_format,
    zip_format, matlab_format)
from nianalysis.interfaces.ants import AntsRegSyn
from nianalysis.interfaces.afni import Tproject
import os
import subprocess as sp
from nianalysis.dataset import Dataset


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
            inputs=[DatasetSpec('ica_dir', directory_format),
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

        fix = pipeline.create_node(finter, name="fix")
        pipeline.connect_input("ica_dir", fix, "feat_dir")
        pipeline.connect_input("train_data", fix, "train_data")
        finter.inputs.component_threshold = pipeline.option(
            'component_threshold')
        finter.inputs.motion_reg = pipeline.option('motion_reg')

        pipeline.connect_output('cleaned_file', fix, 'output')

        pipeline.assert_connected()
        return pipeline

    def optiBET(self, **options):

        optibet = self.create_pipeline(
            name='optiBET',
            inputs=[DatasetSpec('t1', nifti_gz_format)],
            outputs=[DatasetSpec('betted_file', nifti_gz_format),
                     DatasetSpec('betted_mask', nifti_gz_format)],
            description=("python implementation of optiBET.sh"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz',
                             'MNI_template_mask': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain_mask.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        bet1 = optibet.create_node(BET(frac=0.1, reduce_bias=True), name='bet')
        optibet.connect_input('t1', bet1, 'in_file')
        flirt = optibet.create_node(
            FLIRT(out_matrix_file='linear_mat.mat',
                  out_file='linear_reg.nii.gz', searchr_x=[-30, 30],
                  searchr_y=[-30, 30], searchr_z=[-30, 30]), name='flirt')
        flirt.inputs.reference = optibet.option('MNI_template')
        optibet.connect(bet1, 'out_file', flirt, 'in_file')
        fnirt = optibet.create_node(
            FNIRT(config_file='T1_2_MNI152_2mm',
                  fieldcoeff_file='warp_file.nii.gz'), name='fnirt')
        fnirt.inputs.ref_file = optibet.option('MNI_template')
        optibet.connect(flirt, 'out_matrix_file', fnirt, 'affine_file')
        optibet.connect_input('t1', fnirt, 'in_file')
#         optibet.connect_input('ref_file', fnirt, 'ref_file')
        invwarp = optibet.create_node(InvWarp(), name='invwarp')
        optibet.connect(fnirt, 'fieldcoeff_file', invwarp, 'warp')
        optibet.connect_input('t1', invwarp, 'reference')
        applywarp = optibet.create_node(
            ApplyWarp(interp='nn', out_file='warped_file.nii.gz'),
            name='applywarp')
        applywarp.inputs.in_file = optibet.option('MNI_template_mask')
        optibet.connect_input('t1', applywarp, 'ref_file')
#         optibet.connect_input('mask', applywarp, 'in_file')
        optibet.connect(invwarp, 'inverse_warp', applywarp, 'field_file')
        maths1 = optibet.create_node(
            ImageMaths(suffix='_optiBET_brain_mask', op_string='-bin'),
            name='binarize')
        optibet.connect(applywarp, 'out_file', maths1, 'in_file')
        maths2 = optibet.create_node(
            ImageMaths(suffix='_optiBET_brain', op_string='-mas'),
            name='mask')
        optibet.connect_input('t1', maths2, 'in_file')
        optibet.connect(maths1, 'out_file', maths2, 'in_file2')

        optibet.connect_output('betted_mask', maths1, 'out_file')
        optibet.connect_output('betted_file', maths2, 'out_file')

        optibet.assert_connected()
        return optibet

    def ANTsRegistration(self, **options):

        try:
            cmd = 'which ANTS'
            antspath = sp.check_output(cmd, shell=True)
            antspath = '/'.join(antspath.split('/')[0:-1])
            os.environ['ANTSPATH'] = antspath
            print antspath
        except ImportError:
            print "NO ANTs module found. Please ensure to have it in you PATH."

        antsreg = self.create_pipeline(
            name='ANTsReg',
            inputs=[DatasetSpec('betted_file', nifti_gz_format),
                    DatasetSpec('rs_fmri', nifti_gz_format)],
            outputs=[DatasetSpec('epi2T1', nifti_gz_format),
                     DatasetSpec('epi2T1_mat', matlab_format),
                     DatasetSpec('T12MNI_linear', nifti_gz_format),
                     DatasetSpec('T12MNI_mat', matlab_format),
                     DatasetSpec('T12MNI_warp', nifti_gz_format),
                     DatasetSpec('T12MNI_invwarp', nifti_gz_format)],
            description=("python implementation of antsRegistrationSyN.sh"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        bet_rsfmri = antsreg.create_node(BET(), name="bet_rsfmri")
        bet_rsfmri.inputs.robust = True
        bet_rsfmri.inputs.frac = 0.4
        bet_rsfmri.inputs.mask = True
        antsreg.connect_input('rs_fmri', bet_rsfmri, 'in_file')
        epireg = antsreg.create_node(
            AntsRegSyn(num_dimensions=3, transformation='r',
                       out_prefix='epi2T1'), name='ANTsReg')
        antsreg.connect_input('betted_file', epireg, 'ref_file')
        antsreg.connect(bet_rsfmri, 'out_file', epireg, 'input_file')

        t1reg = antsreg.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T12MNI'), name='T1_reg')
        t1reg.inputs.ref_file = antsreg.option('MNI_template')
        antsreg.connect_input('betted_file', t1reg, 'input_file')

        antsreg.connect_output('epi2T1', epireg, 'reg_file')
        antsreg.connect_output('epi2T1_mat', epireg, 'regmat')
        antsreg.connect_output('T12MNI_linear', t1reg, 'reg_file')
        antsreg.connect_output('T12MNI_mat', t1reg, 'regmat')
        antsreg.connect_output('T12MNI_warp', t1reg, 'warp_file')
        antsreg.connect_output('T12MNI_invwarp', t1reg, 'inv_warp')

        antsreg.assert_connected()
        return antsreg

    def MelodicL1(self, **options):

        melodic = self.create_pipeline(
            name='MelodicL1',
            inputs=[DatasetSpec('filtered_file', nifti_gz_format)],
            outputs=[DatasetSpec('melodic_ica', directory_format)],
            description=("python implementation of antsRegistrationSyN.sh"),
            default_options={'brain_thresh_percent': 5},
            version=1,
            citations=[fsl_cite],
            options=options)

        mel = melodic.create_node(MELODIC(), name='fsl-MELODIC')
        mel.inputs.no_bet = True
        mel.inputs.bg_threshold = melodic.option('brain_thresh_percent')
        mel.inputs.tr_sec = 0.754
        mel.inputs.report = True
        mel.inputs.out_stats = True
        mel.inputs.mm_thresh = 0.5
        melodic.connect_input('filtered_data', mel, 'in_files')

        melodic.connect_output('melodic_ica', mel, 'out_dir')

        melodic.assert_connected()
        return melodic

    def rsfMRI_filtering(self, **options):

        pipeline = self.create_pipeline(
            name='rsfMRI_filtering',
            inputs=[DatasetSpec('field_map_mag', nifti_gz_format),
                    DatasetSpec('field_map_phase', nifti_gz_format),
                    DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('rs_fmri', nifti_gz_format),
                    DatasetSpec('betted_file', nifti_gz_format)],
            outputs=[DatasetSpec('filtered_data', nifti_gz_format)],
            description=("Spatial and temporal rsfMRI filtering"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz',
                             'MNI_template_mask': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain_mask.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        bet = pipeline.create_node(BET(), name="bet")
        bet.inputs.robust = True
        pipeline.connect_input('field_map_mag', bet, 'in_file')

        bet_rsfmri = pipeline.create_node(BET(), name="bet_rsfmri")
        bet_rsfmri.inputs.robust = True
        bet_rsfmri.inputs.frac = 0.4
        bet_rsfmri.inputs.mask = True
        pipeline.connect_input('rs_fmri', bet_rsfmri, 'in_file')

        create_fmap = pipeline.create_node(PrepareFieldmap(), name="prepfmap")
        create_fmap.inputs.delta_TE = 2.46
        pipeline.connect(bet, "out_file", create_fmap, "in_magnitude")
        pipeline.connect_input('field_map_phase', create_fmap, 'in_phase')

        fugue = pipeline.create_node(FUGUE(), name='fugue')
        fugue.inputs.unwarp_direction = 'x'
        fugue.inputs.dwell_time = 0.00039
        fugue.inputs.unwarped_file = 'unwarped_rsfmri.nii.gz'
        pipeline.connect(create_fmap, 'out_fieldmap', fugue, 'fmap_in_file')
        pipeline.connect_input('rs_fmri', fugue, 'in_file')

        flirt_t1 = pipeline.create_node(FLIRT(), name='FLIRT_T1')
        flirt_t1.inputs.dof = 6
        flirt_t1.inputs.out_matrix_file = 'example2hires.mat'
        pipeline.connect_input('betted_file', flirt_t1, 'reference')
        pipeline.connect(bet_rsfmri, 'out_file', flirt_t1, 'in_file')

        afni_mc = pipeline.create_node(Volreg(), name='AFNI_MC')
        afni_mc.inputs.zpad = 1
        afni_mc.inputs.out_file = 'rsfmri_mc.nii.gz'
        afni_mc.inputs.oned_file = 'prefiltered_func_data_mcf.par'
        afni_mc.inputs.oned_matrix_save = 'motion_matrices.mat'
        pipeline.connect(fugue, 'unwarped_file', afni_mc, 'in_file')

        filt = pipeline.create_node(Tproject(), name='filtering')
        filt.inputs.stopband = (0, 0.01)
        filt.inputs.delta_t = 0.754
        filt.inputs.polort = 3
        filt.inputs.blur = 3
        pipeline.connect(afni_mc, 'out_file', filt, 'in_file')
        pipeline.connect(bet_rsfmri, 'mask_file', filt, 'mask')

        pipeline.connect_output('filtered_data', filt, 'out_file')

        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('field_map_mag', nifti_gz_format),
        DatasetSpec('field_map_phase', nifti_gz_format),
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('rs_fmri', nifti_gz_format),
        DatasetSpec('melodic_dir', zip_format, feat_pipeline),
        DatasetSpec('train_data', rdata_format),
        DatasetSpec('cleaned_file', nifti_gz_format, fix_pipeline),
        DatasetSpec('betted_file', nifti_gz_format, optiBET),
        DatasetSpec('betted_mask', nifti_gz_format, optiBET),
        DatasetSpec('epi2T1', nifti_gz_format, ANTsRegistration),
        DatasetSpec('epi2T1_mat', matlab_format, ANTsRegistration),
        DatasetSpec('T12MNI_linear', nifti_gz_format, ANTsRegistration),
        DatasetSpec('T12MNI_mat', matlab_format, ANTsRegistration),
        DatasetSpec('T12MNI_warp', nifti_gz_format, ANTsRegistration),
        DatasetSpec('T12MNI_invwarp', nifti_gz_format, ANTsRegistration),
        DatasetSpec('filtered_data', nifti_gz_format, rsfMRI_filtering),
        DatasetSpec('melodic_ica', zip_format, MelodicL1))
