from nianalysis.requirements import fsl5_req, matlab2015_req, ants19_req, mrtrix3_req
from nianalysis.citations import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.data_formats import directory_format, nifti_gz_format, text_matrix_format, csv_format, zip_format
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.qsm import STI, Prepare, FillHoles, QSMSummary
from nianalysis.interfaces import utils
from nianalysis.exceptions import (
    NiAnalysisDatasetNameError, NiAnalysisError, NiAnalysisMissingDatasetError)
from ..base import MRIStudy
from nipype.interfaces import fsl, ants, mrtrix
from nianalysis.interfaces.ants import AntsRegSyn
import os
import subprocess as sp
from nianalysis import pipeline
from nianalysis.pipeline import Pipeline
import nianalysis
from nipype.interfaces.base import traits
import nianalysis.utils
from nipype.interfaces.utility.base import IdentityInterface
#from StdSuites.AppleScript_Suite import space
from scipy.interpolate.interpolate_wrapper import linear

#from nipype.interfaces.fsl.preprocess import (
#    BET, FUGUE, FLIRT, FNIRT, ApplyWarp)
#from nipype.interfaces.afni.preprocess import Volreg, BlurToFWHM
#from nipype.interfaces.fsl.utils import (SwapDimensions, InvWarp, ImageMaths,
#                                         ConvertXFM)
#from nianalysis.interfaces.fsl import (MelodicL1FSF, FSLFIX, CheckLabelFile,
#                                       FSLFixTraining)
#from nipype.interfaces.ants.resampling import ApplyTransforms

class T2StarStudy(MRIStudy):

    def qsm_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Process dual echo data for QSM (TE=[7.38, 22.14])

        NB: Default values come from the STI-Suite
        """
        pipeline = self.create_pipeline(
            name='qsmrecon',
            inputs=[DatasetSpec('raw_coils', directory_format),
                    DatasetSpec('opti_betted_T2s_mask', nifti_gz_format)],
            outputs=[DatasetSpec('qsm', nifti_gz_format),
                     DatasetSpec('tissue_phase', nifti_gz_format),
                     DatasetSpec('tissue_mask', nifti_gz_format)],
            description="Resolve QSM from t2star coils",
            default_options={'qsm_echo_times': [7.38, 22.14]},
            citations=[sti_cites, fsl_cite, matlab_cite],
            version=1,
            options=options)
        
        # Prepare and reformat SWI_COILS
        prepare = pipeline.create_node(interface=Prepare(), name='prepare',
                                       requirements=[matlab2015_req],
                                       wall_time=30, memory=16000)
        pipeline.connect_input('raw_coils', prepare, 'in_dir')
        
        # Phase and QSM for dual echo
        qsmrecon = pipeline.create_node(interface=STI(), name='qsmrecon',
                                        requirements=[matlab2015_req],
                                        wall_time=300, memory=24000)
        qsmrecon.inputs.echo_times = pipeline.option('qsm_echo_times')
        pipeline.connect_input('opti_betted_T2s_mask', qsmrecon, 'mask_file')
        pipeline.connect(prepare,'out_dir', qsmrecon, 'in_dir')
        
        # Use geometry from scanner image
        qsm_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'qsm', qsm_geom, 'dest_file')
        pipeline.connect(prepare,'out_file', qsm_geom, 'in_file')
        
        phase_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_phase_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'tissue_phase', phase_geom, 'dest_file')
        pipeline.connect(prepare,'out_file', phase_geom, 'in_file')
        
        mask_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_mask_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'tissue_mask', mask_geom, 'dest_file')
        pipeline.connect(prepare,'out_file', mask_geom, 'in_file')
        
        # Connect inputs/outputs
        pipeline.connect_output('qsm', qsm_geom, 'out_file')
        pipeline.connect_output('tissue_phase', phase_geom, 'out_file')
        pipeline.connect_output('tissue_mask', mask_geom, 'out_file')

        pipeline.assert_connected()
        return pipeline

# Standalone coil combination code for producing an icerecon space t2s image for registration and segmentation in QSM space
    def prepare_swi_coils(self, **options):
        pipeline = self.create_pipeline(
            name='swi_coils_preparation',
            inputs=[DatasetSpec('raw_coils', directory_format)],
            outputs=[DatasetSpec('t2s', nifti_gz_format)],
            description="Perform preprocessing on raw coils",
            default_options={},
            citations=[matlab_cite],
            version=1,
            options=options)
        
        # Prepare and reformat SWI_COILS for T2s only
        # Prepared output not saved to avoid being uploaded into xnat
        # Only required prior to QSM, so node incorporated into QSM pipeline
        prepare = pipeline.create_node(interface=Prepare(), name='prepare',
                                       requirements=[matlab2015_req],
                                       wall_time=30, memory=16000)
        pipeline.connect_input('raw_coils', prepare, 'in_dir')
        pipeline.connect_output('t2s', prepare,'out_file')
        
        return pipeline

    def optiBET_T1(self, **options):
       
        pipeline = self.create_pipeline(
            name='optiBET_T1',
            inputs=[DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                    DatasetSpec('MNI_to_T1_warp', nifti_gz_format)],
            outputs=[DatasetSpec('opti_betted_T1', nifti_gz_format),
                     DatasetSpec('opti_betted_T1_mask', nifti_gz_format)],
            description=("python implementation of optiBET.sh"),
            default_options={'MNI_template_T1': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz',
                             'MNI_template_mask': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain_mask.nii.gz'},
            version=1,
            citations=[fsl_cite, ants19_req],
            options=options)
        
        fill_holes = pipeline.create_node(interface=FillHoles(), name='fill_holes',
                                       requirements=[matlab2015_req],
                                       wall_time=5, memory=16000)
        fill_holes.inputs.in_file = pipeline.option('MNI_template_mask')
        
        merge_trans = pipeline.create_node(utils.Merge(2), name='merge_transforms')
        pipeline.connect_input('MNI_to_T1_warp', merge_trans, 'in2')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in1')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, False]
        
        pipeline.connect(fill_holes,'out_file', apply_trans, 'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('t1', apply_trans, 'reference_image')
        
        maths1 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_brain_mask', op_string='-bin'),
            name='binarize', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect(apply_trans, 'output_image', maths1, 'in_file')
        
        maths2 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_brain', op_string='-mas'),
            name='mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect_input('t1', maths2, 'in_file')
        pipeline.connect(maths1, 'out_file', maths2, 'in_file2')

        pipeline.connect_output('opti_betted_T1_mask', maths1, 'out_file')
        pipeline.connect_output('opti_betted_T1', maths2, 'out_file')

        pipeline.assert_connected()
        return pipeline
    
    def optiBET_T2s(self, **options):
       
        pipeline = self.create_pipeline(
            name='optiBET_T2',
            inputs=[DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                    DatasetSpec('MNI_to_T1_warp', nifti_gz_format)],
            outputs=[DatasetSpec('opti_betted_T2s', nifti_gz_format),
                     DatasetSpec('opti_betted_T2s_mask', nifti_gz_format)],
            description=("python implementation of optiBET.sh"),
            default_options={'MNI_template_T1': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz',
                             'MNI_template_mask_T2s': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain_mask.nii.gz'},
            version=1,
            citations=[fsl_cite, ants19_req, matlab2015_req],
            options=options)
        
        fill_holes = pipeline.create_node(interface=FillHoles(), name='fill_holes',
                                       requirements=[matlab2015_req],
                                       wall_time=5, memory=16000)
        fill_holes.inputs.in_file = pipeline.option('MNI_template_mask_T2s')
                
        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in2')
        pipeline.connect_input('MNI_to_T1_warp', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, True, False]
        
        pipeline.connect(fill_holes,'out_file', apply_trans, 'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('t2s', apply_trans, 'reference_image')
        
        maths1 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_brain_mask', op_string='-bin'),
            name='binarize', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect(apply_trans, 'output_image', maths1, 'in_file')
        
        maths2 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_brain', op_string='-mas'),
            name='mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect_input('t2s', maths2, 'in_file')
        pipeline.connect(maths1, 'out_file', maths2, 'in_file2')

        pipeline.connect_output('opti_betted_T2s_mask', maths1, 'out_file')
        pipeline.connect_output('opti_betted_T2s', maths2, 'out_file')

        pipeline.assert_connected()
        return pipeline
        
        return pipeline
    
    def bet_T1(self, **options):
        
        pipeline = self.create_pipeline(
            name='BET_T1',
            inputs=[DatasetSpec('t1', nifti_gz_format)],
            outputs=[DatasetSpec('betted_T1', nifti_gz_format),
                     DatasetSpec('betted_T1_mask', nifti_gz_format)],
            description=("python implementation of BET"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        
        bet = pipeline.create_node(
            fsl.BET(frac=0.15, reduce_bias=True), name='bet', requirements=[fsl5_req], memory=8000, wall_time=45)
            
        pipeline.connect_input('t1', bet, 'in_file')
        pipeline.connect_output('betted_T1', bet, 'out_file')
        pipeline.connect_output('betted_T1_mask', bet, 'mask_file')
        
        return pipeline
    
    def bet_T2s(self, **options):
        
        pipeline = self.create_pipeline(
            name='BET_T2s',
            inputs=[DatasetSpec('t2s', nifti_gz_format)],
            outputs=[DatasetSpec('betted_T2s', nifti_gz_format),
                     DatasetSpec('betted_T2s_mask', nifti_gz_format)],
            description=("python implementation of BET"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        
        bet = pipeline.create_node(
            fsl.BET(frac=0.1, reduce_bias=True), name='bet', requirements=[fsl5_req], memory=8000, wall_time=45)
            
        pipeline.connect_input('t2s', bet, 'in_file')
        pipeline.connect_output('betted_T2s', bet, 'out_file')
        pipeline.connect_output('betted_T2s_mask', bet, 'mask_file')
        return pipeline
    
    def linearT2sToT1(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTS_Reg_T2s_to_T1_Mat',
            inputs=[DatasetSpec('betted_T1', nifti_gz_format),
                    DatasetSpec('betted_T2s', nifti_gz_format)],
            outputs=[DatasetSpec('T2s_to_T1_mat', text_matrix_format),
                     DatasetSpec('T2s_in_T1', nifti_gz_format)],
            description=("python implementation of Rigid ANTS Reg for T2s to T1"),           
            default_options={},
            version=1,
            citations=[ants19_req],
            options=options)
                
        t2sreg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='r',
                       out_prefix='T2s2T1'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        pipeline.connect_input('betted_T1', t2sreg, 'ref_file')
        pipeline.connect_input('betted_T2s', t2sreg, 'input_file')
        pipeline.connect_output('T2s_to_T1_mat', t2sreg, 'regmat')
        pipeline.connect_output('T2s_in_T1', t2sreg, 'reg_file')
        
        return pipeline
    
    def nonLinearT1ToMNI(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTS_Reg_T1_to_MNI_Warp',
            inputs=[DatasetSpec('betted_T1', nifti_gz_format)],
            outputs=[DatasetSpec(self._lookup_l_tfm_to_name('MNI'), text_matrix_format),
                     DatasetSpec(self._lookup_nl_tfm_to_name('MNI'), nifti_gz_format),
                     DatasetSpec(self._lookup_nl_tfm_inv_name('MNI'), nifti_gz_format),
                     DatasetSpec('T1_in_MNI', nifti_gz_format)],
            description=("python implementation of Deformable Syn ANTS Reg for T1 to MNI"),           
            default_options={'atlas_template': self._lookup_template_path('MNI')},
            version=1,
            citations=[ants19_req],
            options=options)
                
        t1reg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T1_to_MNI'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        t1reg.inputs.ref_file = pipeline.option('atlas_template')
        
        pipeline.connect_input('betted_T1', t1reg, 'input_file')
        pipeline.connect_output(self._lookup_l_tfm_to_name('MNI'), t1reg, 'regmat')
        pipeline.connect_output(self._lookup_nl_tfm_to_name('MNI'), t1reg, 'warp_file')
        pipeline.connect_output(self._lookup_nl_tfm_inv_name('MNI'), t1reg, 'inv_warp')
        pipeline.connect_output('T1_in_MNI', t1reg, 'reg_file')
        
        return pipeline
    
    def nonLinearT1ToSUIT(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTS_Reg_T1_to_SUIT_Warp',
            inputs=[DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec(self._lookup_l_tfm_to_name('MNI'), text_matrix_format),
                     DatasetSpec(self._lookup_nl_tfm_inv_name('MNI'), nifti_gz_format)],
            outputs=[DatasetSpec(self._lookup_l_tfm_to_name('SUIT'), text_matrix_format),
                     DatasetSpec(self._lookup_nl_tfm_to_name('SUIT'), nifti_gz_format),
                     DatasetSpec(self._lookup_nl_tfm_inv_name('SUIT'), nifti_gz_format),
                     DatasetSpec('T1_in_SUIT', nifti_gz_format),
                     DatasetSpec('SUIT_in_T1', nifti_gz_format)],
            description=("python implementation of Deformable Syn ANTS Reg for T1 to SUIT"),           
            default_options={'SUIT_template': self._lookup_template_path('SUIT')},
            version=1,
            citations=[ants19_req, fsl_cite],
            options=options)
        
        # Initially use MNI space to warp SUIT into T1 and threshold to mask
        merge_trans = pipeline.create_node(utils.Merge(2), name='merge_transforms')
        pipeline.connect_input(self._lookup_nl_tfm_inv_name('MNI'), merge_trans, 'in2')
        pipeline.connect_input(self._lookup_l_tfm_to_name('MNI'), merge_trans, 'in1')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, False]
        apply_trans.inputs.input_image = pipeline.option('SUIT_template')
        
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('t1', apply_trans, 'reference_image')
        
        maths1 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_cerebellum_mask', op_string='-thr 100'),
            name='binarize', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect(apply_trans, 'output_image', maths1, 'in_file')
        
        maths2 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_cerebellum', op_string='-mas'),
            name='mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect_input('t1', maths2, 'in_file')
        pipeline.connect(maths1, 'out_file', maths2, 'in_file2')

        # Use initial SUIT mask to mask out T1 and then register SUIT to T1 only in cerebellum areas
        t1reg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T1_to_SUIT'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        t1reg.inputs.ref_file = pipeline.option('SUIT_template')
        
        # Interpolate SUIT into T1 for QC
        merge_trans_inv = pipeline.create_node(utils.Merge(2), name='merge_transforms_inv')
        pipeline.connect(t1reg, 'inv_warp', merge_trans_inv, 'in2')
        pipeline.connect(t1reg, 'regmat', merge_trans_inv, 'in1')
        
        apply_trans_inv = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform_Inv', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans_inv.inputs.interpolation = 'Linear'
        apply_trans_inv.inputs.input_image_type = 3
        apply_trans_inv.inputs.invert_transform_flags = [True, False]
        apply_trans_inv.inputs.input_image = pipeline.option('SUIT_template')
        apply_trans_inv.inputs.output_image = 'SUIT_in_T1.nii.gz'
        
        pipeline.connect(merge_trans_inv, 'out', apply_trans_inv, 'transforms')
        pipeline.connect_input('t1', apply_trans_inv, 'reference_image')
        
        pipeline.connect(maths2, 'out_file', t1reg, 'input_file')
        pipeline.connect_output(self._lookup_l_tfm_to_name('MNI'), t1reg, 'regmat')
        pipeline.connect_output(self._lookup_nl_tfm_to_name('MNI'), t1reg, 'warp_file')
        pipeline.connect_output(self._lookup_nl_tfm_inv_name('MNI'), t1reg, 'inv_warp')
        pipeline.connect_output('T1_in_SUIT', t1reg, 'reg_file')
        pipeline.connect_output('SUIT_in_T1', apply_trans_inv, 'output_image')
        
        return pipeline    
        
    def qsmInMNI(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform',
            inputs=[DatasetSpec('qsm', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_warp', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('qsm_in_mni', nifti_gz_format)],
            description=("Transform data from T2s to MNI space"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        #cp_geom = pipeline.create_node(fsl.CopyGeom(), name='copy_geomery', requirements=[fsl5_req], memory=8000, wall_time=5)
        #pipeline.connect_input('qsm', cp_geom, 'dest_file')
        #pipeline.connect_input('t2s', cp_geom, 'in_file')
        
        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T1_to_MNI_warp', merge_trans, 'in1')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in2')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.reference_image = pipeline.option('MNI_template')
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        #pipeline.connect(cp_geom, 'out_file', apply_trans, 'input_image')
        pipeline.connect_input('qsm', apply_trans, 'input_image')
        pipeline.connect_output('qsm_in_mni', apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline    

    def qsmInSUIT(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform_SUIT',
            inputs=[DatasetSpec('qsm', nifti_gz_format),
                    DatasetSpec('T1_to_SUIT_warp', nifti_gz_format),
                    DatasetSpec('T1_to_SUIT_mat', text_matrix_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('qsm_in_suit', nifti_gz_format)],
            description=("Transform data from T2s to SUIT space"),
            default_options={'SUIT_template': self._lookup_template_path('SUIT')},
            version=1,
            citations=[fsl_cite],
            options=options)

        #cp_geom = pipeline.create_node(fsl.CopyGeom(), name='copy_geomery', requirements=[fsl5_req], memory=8000, wall_time=5)
        #pipeline.connect_input('qsm', cp_geom, 'dest_file')
        #pipeline.connect_input('t2s', cp_geom, 'in_file')
        
        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T1_to_SUIT_warp', merge_trans, 'in1')
        pipeline.connect_input('T1_to_SUIT_mat', merge_trans, 'in2')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.reference_image = pipeline.option('SUIT_template')
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        #pipeline.connect(cp_geom, 'out_file', apply_trans, 'input_image')
        pipeline.connect_input('qsm', apply_trans, 'input_image')
        pipeline.connect_output('qsm_in_suit', apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline  
            
    def mniInT2s(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform',
            inputs=[DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('MNI_to_T1_warp', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('mni_in_qsm', nifti_gz_format)],
            description=("Transform data from T2s to MNI space"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[ants19_req],
            options=options)

        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in2')
        pipeline.connect_input('MNI_to_T1_warp', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.input_image = pipeline.option('MNI_template')
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, True, False]
        
        pipeline.connect_input('t2s', apply_trans, 'reference_image')
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_output('mni_in_qsm', apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline        
    
    def _lookup_structure_output(self, structure_name):
        outputNames = self._lookup_structure_output_names(structure_name)
        return [DatasetSpec(outputNames[0], nifti_gz_format),DatasetSpec(outputNames[1], nifti_gz_format)]
    
    def _lookup_structure_output_names(self, structure_name):
        if structure_name == 'dentate_nuclei':
            outputNames = ['left_dentate_in_qsm', 'right_dentate_in_qsm']
        elif structure_name == 'caudate':
            outputNames = ['left_caudate_in_qsm', 'right_caudate_in_qsm']
        elif structure_name == 'putamen':
            outputNames = ['left_putamen_in_qsm', 'right_putamen_in_qsm']
        elif structure_name == 'pallidum':
            outputNames = ['left_pallidum_in_qsm', 'right_pallidum_in_qsm']
        elif structure_name == 'thalamus':
            outputNames = ['left_thalamus_in_qsm', 'right_thalamus_in_qsm']
        elif structure_name == 'red_nuclei':
            outputNames = ['left_red_nuclei_in_qsm', 'right_red_nuclei_in_qsm']
        elif structure_name == 'substantia_nigra':
            outputNames = ['left_substantia_nigra_in_qsm', 'right_substantia_nigra_in_qsm']
        else:
            raise NiAnalysisError(
                    "Invalid structure_name in _lookup_structure_output_names: {structure_name}".format(structure_name=structure_name))
        return outputNames
    
    def _lookup_structure_thr(self, structure_name):
        if structure_name in ['dentate_nuclei', 'caudate', 'putamen', 'pallidum', 
                              'thalamus', 'red_nuclei', 'substantia_nigra']:
            thr = '75'
        else:
            raise NiAnalysisError(
                    "Invalid structure_name in _lookup_structure_thr: {structure_name}".format(structure_name=structure_name))
        return thr

    def _lookup_structure_number(self, structure_name): # [LEFT, RIGHT]
        if structure_name == 'dentate_nuclei':
            number = [28, 29]
        elif structure_name == 'caudate':
            number = [4, 15]
        elif structure_name == 'putamen':
            number = [5, 16]
        elif structure_name == 'pallidum':
            number = [6, 17]
        elif structure_name == 'thalamus':
            number = [3, 14]
        elif structure_name == 'red_nuclei':
            number = [0, 2]
        elif structure_name == 'substantia_nigra':
            number = [1, 3]
        else:
            raise NiAnalysisError(
                    "Invalid structure_name in _lookup_structure_number: {structure_name}".format(structure_name=structure_name))
        return number
    
    def _lookup_structure_atlas_path(self, structure_name):
        if structure_name in ['dentate_nuclei']:
            atlas = os.path.abspath(os.path.join(
                os.path.dirname(nianalysis.__file__),
                'atlases','Cerebellum-SUIT-prob.nii'))
        elif structure_name in ['caudate', 'putamen', 'pallidum', 'thalamus']:
            atlas = os.path.abspath(os.path.join(
                os.environ['FSLDIR'],'data','atlases',
                'HarvardOxford','HarvardOxford-sub-prob-2mm.nii.gz'))
        elif structure_name in ['red_nuclei', 'substantia_nigra']:
            atlas = os.path.abspath(os.path.join(
                os.path.dirname(nianalysis.__file__),
                'atlases','ATAG-prob.nii.gz'))
        else:
            raise NiAnalysisError(
                    "Invalid structure_name in _lookup_structure_atlas: {structure_name}".format(structure_name=structure_name))
        return atlas
    
    def _lookup_structure_space(self, structure_name):
        if structure_name == 'dentate_nuclei':
            space = 'SUIT'
        elif structure_name in ['caudate', 'putamen', 'pallidum', 'thalamus']:
            space = 'MNI'
        elif structure_name in ['red_nuclei', 'substantia_nigra']:
            space = 'MNI' #ATAG
        else:
            raise NiAnalysisError(
                    "Invalid structure_name in _lookup_structure_space: {structure_name}".format(structure_name=structure_name))
        return space
    
    def _lookup_template_path(self, space_name):
        if space_name == 'SUIT':
            template_path = os.path.abspath(os.path.join(os.path.dirname(nianalysis.__file__),
                                                          'atlases','SUIT.nii'))
        elif space_name == 'MNI':
            template_path = os.path.abspath(os.path.join(os.environ['FSLDIR'],
                                                         'data','standard','MNI152_T1_2mm_brain.nii.gz'))
        elif space_name == 'ATAG':
            template_path = ''
        else:
            raise NiAnalysisError(
                    "Invalid space_name in _lookup_template_path: {space_name}".format(space_name=space_name))
        return template_path
        
    def _lookup_l_tfm_to_name(self, space_name, isStruct=False):
        if isStruct:
            space_name = self._lookup_structure_space(space_name)
        
        if space_name == 'SUIT':
            inputSpec = 'T1_to_SUIT_mat'
        elif space_name == 'MNI':
            inputSpec = 'T1_to_MNI_mat'
        elif space_name == 'ATAG':
            inputSpec = 'T1_to_ATAG_mat'
        else:
            raise NiAnalysisError(
                    "Invalid space_name in _lookup_l_tfm_to_name: {space_name}".format(space_name=space_name))
        return inputSpec
    
    def _lookup_nl_tfm_inv_name(self, space_name, isStruct=False):
        if isStruct:
            space_name = self._lookup_structure_space(space_name)
        
        if space_name == 'SUIT':
            input_name = 'SUIT_to_T1_warp'
        elif space_name == 'MNI':
            input_name = 'MNI_to_T1_warp'
        elif space_name == 'ATAG':
            input_name = 'ATAG_to_T1_warp'
        else:
            raise NiAnalysisError(
                    "Invalid space_name in _lookup_nl_tfm_inv_name: {space_name}".format(space_name=space_name))
        return input_name
    
    def _lookup_nl_tfm_to_name(self, space_name, isStruct=False):
        if isStruct:
            space_name = self._lookup_structure_space(space_name)
            
        if space_name == 'SUIT':
            input_name = 'T1_to_SUIT_warp'
        elif space_name == 'MNI':
            input_name = 'T1_to_MNI_warp'
        elif space_name == 'ATAG':
            input_name = 'T1_to_ATAG_warp'
        else:
            raise NiAnalysisError(
                    "Invalid space_name in _lookup_nl_tfm_to_names: {space_name}".format(space_name=space_name))
        return input_name
    
    def _mask_pipeline(self, structure_name, **options):
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform_{structure_name}'.format(structure_name=structure_name),
            inputs=[DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format),
                    DatasetSpec(self._lookup_nl_tfm_inv_name(structure_name,True), nifti_gz_format),
                    DatasetSpec(self._lookup_l_tfm_to_name(structure_name,True), text_matrix_format)],
            outputs=self._lookup_structure_output(structure_name),
            description=("Transform {structure_name} atlas to T2s space".format(structure_name=structure_name)),
            default_options={'atlas' : self._lookup_structure_atlas_path(structure_name)},
            version=1,
            citations=[ants19_req],
            options=options)

        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')
        pipeline.connect_input(self._lookup_l_tfm_to_name(structure_name,True), merge_trans, 'in2')
        pipeline.connect_input(self._lookup_nl_tfm_inv_name(structure_name,True), merge_trans, 'in3')
        
        structure_index = self._lookup_structure_number(structure_name)
        
        left_roi = pipeline.create_node(
            fsl.utils.ExtractROI(t_min=structure_index[0], t_size=1),
            name='left_{structure_name}_mask'.format(structure_name=structure_name), 
            requirements=[fsl5_req], memory=16000, wall_time=15)
        left_roi.inputs.in_file = pipeline.option('atlas')
        left_roi.inputs.roi_file = 'left_{structure_name}_mask.nii.gz'.format(structure_name=structure_name)
        
        right_roi = pipeline.create_node(
            fsl.utils.ExtractROI(t_min=structure_index[1], t_size=1),
            name='right_{structure_name}_mask'.format(structure_name=structure_name), 
            requirements=[fsl5_req], memory=16000, wall_time=15)
        right_roi.inputs.in_file = pipeline.option('atlas')
        right_roi.inputs.roi_file = 'right_{structure_name}_mask.nii.gz'.format(structure_name=structure_name)
        
        structure_thr = self._lookup_structure_thr(structure_name)
        
        left_mask = pipeline.create_node(
            fsl.utils.ImageMaths(op_string = '-thr {structure_thr} -bin'.format(structure_thr=structure_thr)),
            name='left_{structure_name}_thr'.format(structure_name=structure_name), 
            requirements=[fsl5_req], memory=8000, wall_time=15)
        pipeline.connect(left_roi,'roi_file', left_mask, 'in_file')
        
        right_mask = pipeline.create_node(
            fsl.utils.ImageMaths(op_string = '-thr {structure_thr} -bin'.format(structure_thr=structure_thr)),
            name='right_{structure_name}_thr'.format(structure_name=structure_name),
            requirements=[fsl5_req], memory=8000, wall_time=15)
        pipeline.connect(right_roi,'roi_file', right_mask, 'in_file')
        
        left_apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), 
            name='ApplyTransform_Left_{structure_name}'.format(structure_name=structure_name),
            requirements=[ants19_req], memory=16000, wall_time=30)
        left_apply_trans.inputs.interpolation = 'NearestNeighbor'
        left_apply_trans.inputs.input_image_type = 3
        left_apply_trans.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect(left_mask,'out_file',left_apply_trans,'input_image')
        
        right_apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(),
             name='ApplyTransform_Right_{structure_name}'.format(structure_name=structure_name),
             requirements=[ants19_req], memory=16000, wall_time=30)
        right_apply_trans.inputs.interpolation = 'NearestNeighbor'
        right_apply_trans.inputs.input_image_type = 3
        right_apply_trans.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect(right_mask,'out_file',right_apply_trans,'input_image')
        
        output_names = self._lookup_structure_output_names(structure_name)
        
        pipeline.connect_input('t2s', left_apply_trans, 'reference_image')
        pipeline.connect(merge_trans, 'out', left_apply_trans, 'transforms')
        pipeline.connect_output(output_names[0], left_apply_trans, 'output_image')
        
        pipeline.connect_input('t2s', right_apply_trans, 'reference_image')
        pipeline.connect(merge_trans, 'out', right_apply_trans, 'transforms')
        pipeline.connect_output(output_names[1], right_apply_trans, 'output_image')

        pipeline.assert_connected()
        
        return pipeline
            
    def dentate_masks(self, **options):
        return self._mask_pipeline('dentate_nuclei')
    
    def caudate_masks(self, **options):
        return self._mask_pipeline('caudate')
    
    def putamen_masks(self, **options):
        return self._mask_pipeline('putamen')
    
    def pallidum_masks(self, **options):
        return self._mask_pipeline('pallidum')
    
    def thalamus_masks(self, **options):
        return self._mask_pipeline('thalamus')
    
    def red_nuclei_masks(self, **options):
        return self._mask_pipeline('red_nuclei')
    
    def substantia_nigra_masks(self, **options):
        return self._mask_pipeline('substantia_nigra')
    
    def _lookup_study_structures(self, study_name):
        if study_name in ['ASPREE', 'FRDA']:
            structure_list = ['dentate_nuclei', 'caudate','putamen','pallidum','thalamus','red_nuclei','substantia_nigra']
        else:
            raise NiAnalysisError(
                    "Invalid study_name in _lookup_study_structures: {study_name}".format(study_name=study_name))
        return structure_list
        
    def analysis_pipeline(self, **options):
        
        # Build list of inputs before creating the pipeline (Tom said so!)
        # Cannot use pipeline options and pipeline default for 'study_name'
        input_list = [DatasetSpec('qsm', nifti_gz_format)]
        for structure_name in self._lookup_study_structures(options.get('study_name','FRDA')):
            input_list.extend(self._lookup_structure_output(structure_name))
        
        pipeline = self.create_pipeline(
            name='QSM_Analysis',
            inputs=input_list,
            outputs=[DatasetSpec('qsm_summary', csv_format)],
            default_options={'study_name' : 'FRDA'},
            description=("Regional statistics of QSM images."),
            version=1,
            citations=[ants19_req, fsl5_req],
            options=options)
        
        # Build list of fields for summary
        field_list = [] #['in_subject_id','in_visit_id']
        for structure_name in self._lookup_study_structures(pipeline.option('study_name')):
            field_list.extend(['in_left_{structure_name}_mean'.format(structure_name=structure_name), 
                               'in_left_{structure_name}_std'.format(structure_name=structure_name),
                               'in_right_{structure_name}_mean'.format(structure_name=structure_name), 
                               'in_right_{structure_name}_std'.format(structure_name=structure_name)])
            
        merge_stats = pipeline.create_node(
            interface=utils.Merge(2*len(field_list)), 
            name='merge_stats_{study_name}'.format(study_name=pipeline.option('study_name')),
            wall_time=60, 
            memory=4000)
            
        # Create the mean and standard deviation nodes for left and right of each structure
        for i, structure_name in enumerate(self._lookup_study_structures(pipeline.option('study_name'))):
            mask_names = self._lookup_structure_output_names(structure_name)
               
            right_apply_mask_mean = pipeline.create_node(fsl.ImageStats(),
                                                         name='Stats_Right_Mean_{structure_name}'.format(structure_name=structure_name),
                                                         requirements=[fsl5_req], memory=4000, wall_time=15)
            right_apply_mask_mean.inputs.op_string = '-k %s -m'        
            pipeline.connect_input('qsm', right_apply_mask_mean, 'in_file')
            pipeline.connect_input(mask_names[1], right_apply_mask_mean, 'mask_file')
            pipeline.connect(right_apply_mask_mean, 'out_stat', merge_stats, 'in'+str(4*i+1))
        
            right_apply_mask_std = pipeline.create_node(fsl.ImageStats(),
                                                         name='Stats_Right_Std_{structure_name}'.format(structure_name=structure_name),
                                                         requirements=[fsl5_req], memory=4000, wall_time=15)
            right_apply_mask_std.inputs.op_string = '-k %s -s'        
            pipeline.connect_input('qsm', right_apply_mask_std, 'in_file')
            pipeline.connect_input(mask_names[1], right_apply_mask_std, 'mask_file')
            pipeline.connect(right_apply_mask_std, 'out_stat', merge_stats, 'in'+str(4*i+2))
        
            left_apply_mask_mean = pipeline.create_node(fsl.ImageStats(),
                                                         name='Stats_Left_Mean_{structure_name}'.format(structure_name=structure_name),
                                                         requirements=[fsl5_req], memory=4000, wall_time=15)
            left_apply_mask_mean.inputs.op_string = '-k %s -m'        
            pipeline.connect_input('qsm', left_apply_mask_mean, 'in_file')
            pipeline.connect_input(mask_names[0], left_apply_mask_mean, 'mask_file')
            pipeline.connect(left_apply_mask_mean, 'out_stat', merge_stats, 'in'+str(4*i+3))
        
            left_apply_mask_std = pipeline.create_node(fsl.ImageStats(),
                                                         name='Stats_Left_Std_{structure_name}'.format(structure_name=structure_name),
                                                         requirements=[fsl5_req], memory=4000, wall_time=15)
            left_apply_mask_std.inputs.op_string = '-k %s -s'        
            pipeline.connect_input('qsm', left_apply_mask_std, 'in_file')
            pipeline.connect_input(mask_names[0], left_apply_mask_std, 'mask_file')
            pipeline.connect(left_apply_mask_std, 'out_stat', merge_stats, 'in'+str(4*i+4))
        
        identity_node = pipeline.create_join_subjects_node(interface=IdentityInterface(['in_subject_id','in_visit_id','in_field_values']),
                                                         name='Join_Subjects_Identity',
                                                         joinfield=['in_subject_id','in_visit_id','in_field_values'],
                                                         wall_time=60, memory=4000)
        
        pipeline.connect(merge_stats, 'out', identity_node, 'in_field_values')
        pipeline.connect_subject_id(identity_node, 'in_subject_id')
        pipeline.connect_visit_id(identity_node,'in_visit_id')
        
        summarise_results = pipeline.create_join_visits_node(
            interface=QSMSummary(), 
            name='summarise_qsm',
            joinfield=['in_subject_id','in_visit_id','in_field_values'],
            wall_time=60, 
            memory=4000)
        summarise_results.inputs.in_field_names = field_list
        pipeline.connect(identity_node, 'in_field_values', summarise_results, 'in_field_values')
        pipeline.connect(identity_node, 'in_subject_id', summarise_results, 'in_subject_id')
        pipeline.connect(identity_node, 'in_visit_id', summarise_results, 'in_visit_id')
        
        pipeline.connect_output('qsm_summary', summarise_results, 'out_file')
        
        return pipeline
    
    
    _dataset_specs = set_dataset_specs(
        
        # Primary inputs
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('raw_coils', zip_format,
                    description=("Reconstructed T2* complex image for each "
                                 "coil without standardisation.")),
                                       
        DatasetSpec('t2s', nifti_gz_format, prepare_swi_coils),
                          
        # Brain extraction and bias correction                                   
        DatasetSpec('betted_T1', nifti_gz_format, bet_T1), 
        DatasetSpec('betted_T1_mask', nifti_gz_format, bet_T1),   
             
        DatasetSpec('betted_T2s', nifti_gz_format, bet_T2s),     
        DatasetSpec('betted_T2s_mask', nifti_gz_format, bet_T2s),
        
        DatasetSpec('opti_betted_T1', nifti_gz_format, optiBET_T1),
        DatasetSpec('opti_betted_T1_mask', nifti_gz_format, optiBET_T1),
        
        DatasetSpec('opti_betted_T2s', nifti_gz_format, optiBET_T2s),
        DatasetSpec('opti_betted_T2s_mask', nifti_gz_format, optiBET_T2s),
        
        # Transformation between contrasts in subject space
        DatasetSpec('T2s_to_T1_mat', text_matrix_format, linearT2sToT1),
        DatasetSpec('T2s_in_T1', nifti_gz_format, linearT2sToT1),
        
        # Transformation into standard MNI space
        DatasetSpec('T1_to_MNI_mat', text_matrix_format, nonLinearT1ToMNI),
        DatasetSpec('T1_to_MNI_warp', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('MNI_to_T1_warp', nifti_gz_format, nonLinearT1ToMNI),
        
        DatasetSpec('T1_in_MNI', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('MNI_in_T1', nifti_gz_format, nonLinearT1ToMNI),
        
        # Transformation into standard SUIT space
        DatasetSpec('T1_to_SUIT_mat', text_matrix_format, nonLinearT1ToSUIT),
        DatasetSpec('T1_to_SUIT_warp', nifti_gz_format, nonLinearT1ToSUIT),
        DatasetSpec('SUIT_to_T1_warp', nifti_gz_format, nonLinearT1ToSUIT),
        
        DatasetSpec('T1_in_SUIT', nifti_gz_format, nonLinearT1ToSUIT),
        DatasetSpec('SUIT_in_T1', nifti_gz_format, nonLinearT1ToSUIT),
                                
        # QSM and phase processing                        
        DatasetSpec('qsm', nifti_gz_format, qsm_pipeline,
                    description=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        DatasetSpec('tissue_phase', nifti_gz_format, qsm_pipeline,
                    description=("Phase map for each coil following unwrapping"
                                 " and background field removal")),
        DatasetSpec('tissue_mask', nifti_gz_format, qsm_pipeline,
                    description=("Mask for each coil corresponding to areas of"
                                 " high magnitude")),
        
        # Data for analysis in MNI space (and quality control)                                   
        DatasetSpec('qsm_in_mni', nifti_gz_format, qsmInMNI),
        DatasetSpec('mni_in_qsm', nifti_gz_format, mniInT2s),
        
        # Data for analysis in SUIT space (and quality control)                                   
        DatasetSpec('qsm_in_suit', nifti_gz_format, qsmInSUIT),
        
        # Masks for analysis in subject space
        DatasetSpec('left_dentate_in_qsm', nifti_gz_format, dentate_masks),
        DatasetSpec('right_dentate_in_qsm', nifti_gz_format, dentate_masks),
        DatasetSpec('left_red_nuclei_in_qsm', nifti_gz_format, red_nuclei_masks),
        DatasetSpec('right_red_nuclei_in_qsm', nifti_gz_format, red_nuclei_masks),
        DatasetSpec('left_substantia_nigra_in_qsm', nifti_gz_format, substantia_nigra_masks),
        DatasetSpec('right_substantia_nigra_in_qsm', nifti_gz_format, substantia_nigra_masks),
        DatasetSpec('left_pallidum_in_qsm', nifti_gz_format, pallidum_masks),
        DatasetSpec('right_pallidum_in_qsm', nifti_gz_format, pallidum_masks),
        DatasetSpec('left_thalamus_in_qsm', nifti_gz_format, thalamus_masks),
        DatasetSpec('right_thalamus_in_qsm', nifti_gz_format, thalamus_masks),
        DatasetSpec('left_putamen_in_qsm', nifti_gz_format, putamen_masks),
        DatasetSpec('right_putamen_in_qsm', nifti_gz_format, putamen_masks),
        DatasetSpec('left_caudate_in_qsm', nifti_gz_format, caudate_masks),
        DatasetSpec('right_caudate_in_qsm', nifti_gz_format, caudate_masks),
    
        # Study-specific analysis summary files
        DatasetSpec('qsm_summary', csv_format, analysis_pipeline, multiplicity='per_project'))