from nianalysis.requirements import fsl5_req, matlab2015_req, ants19_req, mrtrix3_req
from nianalysis.citations import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.data_formats import directory_format, nifti_gz_format, text_matrix_format, csv_format
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.qsm import STI, Prepare, FillHoles, CSVSummary
from nianalysis.interfaces import utils
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
            inputs=[DatasetSpec('prepared_coils', directory_format),
                    DatasetSpec('opti_betted_T2s_mask', nifti_gz_format),
                    DatasetSpec('t2s', nifti_gz_format)],
            # TODO should this be primary?
            outputs=[DatasetSpec('qsm', nifti_gz_format),
                     DatasetSpec('tissue_phase', nifti_gz_format),
                     DatasetSpec('tissue_mask', nifti_gz_format)],
            description="Resolve QSM from t2star coils",
            default_options={'qsm_echo_times': [7.38, 22.14]},
            citations=[sti_cites, fsl_cite, matlab_cite],
            version=1,
            options=options)

        # Phase and QSM for dual echo
        qsmrecon = pipeline.create_node(interface=STI(), name='qsmrecon',
                                        requirements=[matlab2015_req],
                                        wall_time=300, memory=24000)
        qsmrecon.inputs.echo_times = pipeline.option('qsm_echo_times')
        pipeline.connect_input('opti_betted_T2s_mask', qsmrecon, 'mask_file')
        pipeline.connect_input('prepared_coils', qsmrecon, 'in_dir')
        
        # Use geometry from scanner image
        qsm_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'qsm', qsm_geom, 'dest_file')
        pipeline.connect_input('t2s', qsm_geom, 'in_file')
        
        phase_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_phase_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'tissue_phase', phase_geom, 'dest_file')
        pipeline.connect_input('t2s', phase_geom, 'in_file')
        
        mask_geom = pipeline.create_node(fsl.CopyGeom(), name='qsm_mask_copy_geomery', requirements=[fsl5_req], memory=4000, wall_time=5)
        pipeline.connect(qsmrecon, 'tissue_mask', mask_geom, 'dest_file')
        pipeline.connect_input('t2s', mask_geom, 'in_file')
        
        # Connect inputs/outputs
        pipeline.connect_output('qsm', qsm_geom, 'out_file')
        pipeline.connect_output('tissue_phase', phase_geom, 'out_file')
        pipeline.connect_output('tissue_mask', mask_geom, 'out_file')

        pipeline.assert_connected()
        return pipeline

    def prepare_swi_coils(self, **options):
        pipeline = self.create_pipeline(
            name='swi_coils_preparation',
            inputs=[DatasetSpec('raw_coils', directory_format)],
            outputs=[DatasetSpec('prepared_coils', directory_format),
                     DatasetSpec('t2s', nifti_gz_format)],
            description="Perform preprocessing on raw coils",
            default_options={},
            citations=[matlab_cite],
            version=1,
            options=options)
        
        # Prepare and reformat SWI_COILS
        prepare = pipeline.create_node(interface=Prepare(), name='prepare',
                                       requirements=[matlab2015_req],
                                       wall_time=30, memory=16000)
        pipeline.connect_input('raw_coils', prepare, 'in_dir')
        pipeline.connect_output('prepared_coils', prepare,'out_dir')
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
            outputs=[DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                     DatasetSpec('T1_to_MNI_warp', nifti_gz_format),
                     DatasetSpec('MNI_to_T1_warp', nifti_gz_format),
                     DatasetSpec('T1_in_MNI', nifti_gz_format)],
            description=("python implementation of Deformable Syn ANTS Reg for T1 to MNI"),           
            default_options={'MNI_template_T1': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[ants19_req],
            options=options)
                
        t1reg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T1_to_MNI'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        t1reg.inputs.ref_file = pipeline.option('MNI_template_T1')
        
        pipeline.connect_input('betted_T1', t1reg, 'input_file')
        pipeline.connect_output('T1_to_MNI_mat', t1reg, 'regmat')
        pipeline.connect_output('T1_to_MNI_warp', t1reg, 'warp_file')
        pipeline.connect_output('MNI_to_T1_warp', t1reg, 'inv_warp')
        pipeline.connect_output('T1_in_MNI', t1reg, 'reg_file')
        
        return pipeline
    
    def nonLinearT1ToSUIT(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTS_Reg_T1_to_SUIT_Warp',
            inputs=[DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                     DatasetSpec('MNI_to_T1_warp', nifti_gz_format)],
            outputs=[DatasetSpec('T1_to_SUIT_mat', text_matrix_format),
                     DatasetSpec('T1_to_SUIT_warp', nifti_gz_format),
                     DatasetSpec('SUIT_to_T1_warp', nifti_gz_format),
                     DatasetSpec('T1_in_SUIT', nifti_gz_format),
                     DatasetSpec('SUIT_in_T1', nifti_gz_format)],
            description=("python implementation of Deformable Syn ANTS Reg for T1 to SUIT"),           
            default_options={'SUIT_template': os.path.abspath(os.path.join(os.path.dirname(nianalysis.__file__),
                                                          'atlases','SUIT.nii'))},
            version=1,
            citations=[ants19_req, fsl_cite],
            options=options)
        
        # Initially use MNI space to warp SUIT into T1 and threshold to mask
        merge_trans = pipeline.create_node(utils.Merge(2), name='merge_transforms')
        pipeline.connect_input('MNI_to_T1_warp', merge_trans, 'in2')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in1')

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
        pipeline.connect_output('T1_to_SUIT_mat', t1reg, 'regmat')
        pipeline.connect_output('T1_to_SUIT_warp', t1reg, 'warp_file')
        pipeline.connect_output('SUIT_to_T1_warp', t1reg, 'inv_warp')
        pipeline.connect_output('T1_in_SUIT', t1reg, 'reg_file')
        pipeline.connect_output('SUIT_in_T1', apply_trans_inv, 'output_image')
        
        return pipeline    
        
        
    def qsmInMNI(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform',
            inputs=[DatasetSpec('qsm', nifti_gz_format),
                    DatasetSpec('t2s', nifti_gz_format),
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
  
    def frda_masks(self, **options):
        
        return pipeline
    
    def aspree_masks(self, **options):
        
        return pipeline
    
    def dentate_masks(self, **options):
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform_Dentate',
            inputs=[DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('SUIT_to_T1_warp', nifti_gz_format),
                    DatasetSpec('T1_to_SUIT_mat', text_matrix_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('left_dentate_in_qsm', nifti_gz_format),
                     DatasetSpec('right_dentate_in_qsm', nifti_gz_format)],
            description=("Transform dentate atlases from MNI to T2s space"),
            default_options={'SUIT_prob' :  
                             os.path.abspath(os.path.join(os.path.dirname(nianalysis.__file__),
                                                          'atlases','Cerebellum-SUIT-prob.nii'))},
            version=1,
            citations=[ants19_req],
            options=options)

        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')
        pipeline.connect_input('T1_to_SUIT_mat', merge_trans, 'in2')
        pipeline.connect_input('SUIT_to_T1_warp', merge_trans, 'in3')
        
        left_roi = pipeline.create_node(
            fsl.utils.ExtractROI(t_min=28, t_size=1),
            name='left_dn_mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        left_roi.inputs.in_file = pipeline.option('SUIT_prob')
        left_roi.inputs.roi_file = 'left_dn_mask.nii.gz'
        
        left_mask = pipeline.create_node(
            fsl.utils.ImageMaths(op_string = '-thr 50 -bin'),
            name='left_dn_thr', requirements=[fsl5_req], memory=8000, wall_time=5)
        pipeline.connect(left_roi,'roi_file', left_mask, 'in_file')

        left_apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform_Left', requirements=[ants19_req], memory=16000, wall_time=30)
        #left_apply_trans.inputs.input_image = pipeline.option('left_dentate_nucleus_template')
        left_apply_trans.inputs.interpolation = 'NearestNeighbor'
        left_apply_trans.inputs.input_image_type = 3
        left_apply_trans.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect(left_mask,'out_file',left_apply_trans,'input_image')
        
        pipeline.connect_input('t2s', left_apply_trans, 'reference_image')
        pipeline.connect(merge_trans, 'out', left_apply_trans, 'transforms')
        pipeline.connect_output('left_dentate_in_qsm', left_apply_trans, 'output_image')

        right_roi = pipeline.create_node(
            fsl.utils.ExtractROI(t_min=29, t_size=1),
            name='right_dn_mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        right_roi.inputs.in_file = pipeline.option('SUIT_prob')
        right_roi.inputs.roi_file = 'right_dn_mask.nii.gz'
        
        right_mask = pipeline.create_node(
            fsl.utils.ImageMaths(op_string = '-thr 50 -bin'),
            name='right_dn_thr', requirements=[fsl5_req], memory=8000, wall_time=5)
        pipeline.connect(right_roi,'roi_file', right_mask, 'in_file')
        
        right_apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform_Right', requirements=[ants19_req], memory=16000, wall_time=30)
        #left_apply_trans.inputs.input_image = pipeline.option('left_dentate_nucleus_template')
        right_apply_trans.inputs.interpolation = 'NearestNeighbor'
        right_apply_trans.inputs.input_image_type = 3
        right_apply_trans.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect(right_mask,'out_file',right_apply_trans,'input_image')
        
        pipeline.connect_input('t2s', right_apply_trans, 'reference_image')
        pipeline.connect(merge_trans, 'out', right_apply_trans, 'transforms')
        pipeline.connect_output('right_dentate_in_qsm', right_apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline
    
    def dentate_analysis(self, **options):
        
        pipeline = self.create_pipeline(
            name='MaskQSM_Dentate',
            inputs=[DatasetSpec('qsm', nifti_gz_format),
                    DatasetSpec('right_dentate_in_qsm', nifti_gz_format),
                    DatasetSpec('left_dentate_in_qsm', nifti_gz_format)],
            outputs=[DatasetSpec('qsm_summary', csv_format)],
                    #DatasetSpec('left_dentate_qsm', traits.Str),
                    #DatasetSpec('right_dentate_qsm', traits.Str),
            description=("Mask out QSM based on dentate regions"),
            default_options={},
            version=1,
            citations=[ants19_req],
            options=options)
            
        right_apply_mask_mean = pipeline.create_node(fsl.ImageStats(), name='Stats_Right_Mean',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        right_apply_mask_mean.inputs.op_string = '-k %s -m'        
        pipeline.connect_input('qsm', right_apply_mask_mean, 'in_file')
        pipeline.connect_input('right_dentate_in_qsm', right_apply_mask_mean, 'mask_file')
        
        
        right_apply_mask_std = pipeline.create_node(fsl.ImageStats(), name='Stats_Right_Std',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        right_apply_mask_std.inputs.op_string = '-k %s -s'        
        pipeline.connect_input('qsm', right_apply_mask_std, 'in_file')
        pipeline.connect_input('right_dentate_in_qsm', right_apply_mask_std, 'mask_file')
        
        right_apply_mask_hist = pipeline.create_node(fsl.ImageStats(), name='Stats_Right_Hist',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        right_apply_mask_hist.inputs.op_string = '-k %s -h 5'        
        pipeline.connect_input('qsm', right_apply_mask_hist, 'in_file')
        pipeline.connect_input('right_dentate_in_qsm', right_apply_mask_hist, 'mask_file')

        left_apply_mask_mean = pipeline.create_node(fsl.ImageStats(), name='Stats_Left_Mean',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        left_apply_mask_mean.inputs.op_string = '-k %s -m'        
        pipeline.connect_input('qsm', left_apply_mask_mean, 'in_file')
        pipeline.connect_input('left_dentate_in_qsm', left_apply_mask_mean, 'mask_file')
        
        
        left_apply_mask_std = pipeline.create_node(fsl.ImageStats(), name='Stats_Left_Std',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        left_apply_mask_std.inputs.op_string = '-k %s -s'        
        pipeline.connect_input('qsm', left_apply_mask_std, 'in_file')
        pipeline.connect_input('left_dentate_in_qsm', left_apply_mask_std, 'mask_file')
        
        left_apply_mask_hist = pipeline.create_node(fsl.ImageStats(), name='Stats_Left_Hist',
                                                requirements=[fsl5_req], memory=4000, wall_time=5)
        left_apply_mask_hist.inputs.op_string = '-k %s -h 5'        
        pipeline.connect_input('qsm', left_apply_mask_hist, 'in_file')
        pipeline.connect_input('left_dentate_in_qsm', left_apply_mask_hist, 'mask_file')
        
        summarise_results = pipeline.create_join_subjects_node(
            interface=CSVSummary(), 
            joinfield=['in_ldn_mean','in_ldn_std','in_ldn_hist','in_rdn_mean','in_rdn_std','in_rdn_hist'],
            name='summarise_qsm', wall_time=60, memory=4000)
        pipeline.connect(left_apply_mask_mean,'out_stat',summarise_results, 'in_ldn_mean')
        pipeline.connect(left_apply_mask_std,'out_stat',summarise_results, 'in_ldn_std')
        pipeline.connect(left_apply_mask_hist,'out_stat',summarise_results, 'in_ldn_hist')        
        pipeline.connect(right_apply_mask_mean,'out_stat',summarise_results, 'in_rdn_mean')
        pipeline.connect(right_apply_mask_std,'out_stat',summarise_results, 'in_rdn_std')
        pipeline.connect(right_apply_mask_hist,'out_stat',summarise_results, 'in_rdn_hist')
        
        pipeline.connect_output('qsm_summary', summarise_results, 'out_file')
        
        return pipeline
    
    def red_nuclei_masks(self, **options):
        
        return pipeline
    
    def subcortical_structure_masks(self, **options):
        
        return pipeline
    
    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('raw_coils', directory_format,
                    description=("Reconstructed T2* complex image for each "
                                 "coil without standardisation.")),
                                       
        DatasetSpec('prepared_coils', directory_format, prepare_swi_coils),
        DatasetSpec('t2s', nifti_gz_format, prepare_swi_coils),
                                           
        DatasetSpec('betted_T1', nifti_gz_format, bet_T1), 
        DatasetSpec('betted_T1_mask', nifti_gz_format, bet_T1),   
             
        DatasetSpec('betted_T2s', nifti_gz_format, bet_T2s),     
        DatasetSpec('betted_T2s_mask', nifti_gz_format, bet_T2s),
        
        DatasetSpec('opti_betted_T1', nifti_gz_format, optiBET_T1),
        DatasetSpec('opti_betted_T1_mask', nifti_gz_format, optiBET_T1),
        
        DatasetSpec('opti_betted_T2s', nifti_gz_format, optiBET_T2s),
        DatasetSpec('opti_betted_T2s_mask', nifti_gz_format, optiBET_T2s),
        
        DatasetSpec('T2s_to_T1_mat', text_matrix_format, linearT2sToT1),
        DatasetSpec('T2s_in_T1', nifti_gz_format, linearT2sToT1),
        
        DatasetSpec('T1_to_MNI_mat', text_matrix_format, nonLinearT1ToMNI),
        DatasetSpec('T1_to_MNI_warp', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('MNI_to_T1_warp', nifti_gz_format, nonLinearT1ToMNI),
        
        DatasetSpec('T1_in_MNI', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('MNI_in_T1', nifti_gz_format, nonLinearT1ToMNI),
        
        DatasetSpec('T1_to_SUIT_mat', text_matrix_format, nonLinearT1ToSUIT),
        DatasetSpec('T1_to_SUIT_warp', nifti_gz_format, nonLinearT1ToSUIT),
        DatasetSpec('SUIT_to_T1_warp', nifti_gz_format, nonLinearT1ToSUIT),
        
        DatasetSpec('T1_in_SUIT', nifti_gz_format, nonLinearT1ToSUIT),
        DatasetSpec('SUIT_in_T1', nifti_gz_format, nonLinearT1ToSUIT),
                                
        DatasetSpec('qsm', nifti_gz_format, qsm_pipeline,
                    description=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        DatasetSpec('tissue_phase', nifti_gz_format, qsm_pipeline,
                    description=("Phase map for each coil following unwrapping"
                                 " and background field removal")),
        DatasetSpec('tissue_mask', nifti_gz_format, qsm_pipeline,
                    description=("Mask for each coil corresponding to areas of"
                                 " high magnitude")),
                                           
        DatasetSpec('qsm_in_mni', nifti_gz_format, qsmInMNI),
        DatasetSpec('mni_in_qsm', nifti_gz_format, qsmInMNI),
        
        DatasetSpec('left_dentate_in_qsm', nifti_gz_format, dentate_masks),
        DatasetSpec('right_dentate_in_qsm', nifti_gz_format, dentate_masks),
        DatasetSpec('left_red_nuclei_in_qsm', nifti_gz_format, red_nuclei_masks),
        DatasetSpec('right_red_nuclei_in_qsm', nifti_gz_format, red_nuclei_masks),
        DatasetSpec('left_substantia_nigra_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('right_substantia_nigra_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('left_globus_pallidus_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('right_globus_pallidus_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('left_thalamus_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('right_thalamus_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('left_putamen_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('right_putamen_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('left_caudate_in_qsm', nifti_gz_format, subcortical_structure_masks),
        DatasetSpec('right_caudate_in_qsm', nifti_gz_format, subcortical_structure_masks),
    
        DatasetSpec('qsm_summary', csv_format, dentate_analysis))
    

''' Deprecated (to be removed in future versions)          
        # legacy
        DatasetSpec('qsm_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        DatasetSpec('tissue_phase_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Phase map for each coil following unwrapping"
                                 " and background field removal")),
        DatasetSpec('tissue_mask_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Mask for each coil corresponding to areas of"
                                 " high magnitude")),
        DatasetSpec('qsm_mask_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Brain mask generated from T2* image"))
        )
'''
    
''' Deprecated (to be removed in future versions)  
    def ANTsRegistration(self, **options):

#        try:
#            cmd = 'which ANTS'
#            antspath = sp.check_output(cmd, shell=True)
#            antspath = '/'.join(antspath.split('/')[0:-1])
#            os.environ['ANTSPATH'] = antspath
#            print antspath
#        except ImportError:
#            print "NO ANTs module found. Please ensure to have it in you PATH."

        antsreg = self.create_pipeline(
            name='ANTsReg',
            inputs=[DatasetSpec('betted_file', nifti_gz_format),
                    DatasetSpec('t2s', nifti_gz_format)],
            outputs=[DatasetSpec('T2s2T1', nifti_gz_format),
                     DatasetSpec('T2s2T1_mat', text_matrix_format),
                     DatasetSpec('T12MNI_linear', nifti_gz_format),
                     DatasetSpec('T12MNI_mat', text_matrix_format),
                     DatasetSpec('T12MNI_warp', nifti_gz_format),
                     DatasetSpec('T12MNI_invwarp', nifti_gz_format)],
            description=("python implementation of antsRegistrationSyN.sh"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        bet_t2s = antsreg.create_node(fsl.BET(), name="bet_t2s", requirements=[fsl5_req], memory=16000, wall_time=30)
        bet_t2s.inputs.robust = True
        bet_t2s.inputs.frac = 0.4
        bet_t2s.inputs.mask = True
        antsreg.connect_input('t2s', bet_t2s, 'in_file')
        t2sreg = antsreg.create_node(
            AntsRegSyn(num_dimensions=3, transformation='r',
                       out_prefix='T2s2T1'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        antsreg.connect_input('betted_file', t2sreg, 'ref_file')
        antsreg.connect(bet_t2s, 'out_file', t2sreg, 'input_file')

        t1reg = antsreg.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T12MNI'), name='T1_reg', requirements=[ants19_req], memory=16000, wall_time=30)
        t1reg.inputs.ref_file = antsreg.option('MNI_template')
        antsreg.connect_input('betted_file', t1reg, 'input_file')

        antsreg.connect_output('T2s2T1', t2sreg, 'reg_file')
        antsreg.connect_output('T2s2T1_mat', t2sreg, 'regmat')
        antsreg.connect_output('T12MNI_linear', t1reg, 'reg_file')
        antsreg.connect_output('T12MNI_mat', t1reg, 'regmat')
        antsreg.connect_output('T12MNI_warp', t1reg, 'warp_file')
        antsreg.connect_output('T12MNI_invwarp', t1reg, 'inv_warp')

        antsreg.assert_connected()
        return antsreg
'''

''' Deprecated (to be removed in future versions)    
    def applyTransform(self, **options):

        pipeline = self.create_pipeline(
            name='ANTsApplyTransform',
            inputs=[DatasetSpec('qsm', nifti_gz_format),
                    DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('T12MNI_warp', nifti_gz_format),
                    DatasetSpec('T12MNI_mat', text_matrix_format),
                    DatasetSpec('T2s2T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('qsm_in_mni', nifti_gz_format)],
            description=("Transform data from T2s to MNI space"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[fsl_cite],
            options=options)

        cp_geom = pipeline.create_node(fsl.CopyGeom(), name='copy_geomery', requirements=[fsl5_req], memory=8000, wall_time=5)
        pipeline.connect_input('qsm', cp_geom, 'dest_file')
        pipeline.connect_input('t2s', cp_geom, 'in_file')
        
        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T12MNI_warp', merge_trans, 'in1')
        pipeline.connect_input('T12MNI_mat', merge_trans, 'in2')
        pipeline.connect_input('T2s2T1_mat', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.reference_image = pipeline.option('MNI_template')
#         apply_trans.inputs.dimension = 3
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect(cp_geom, 'out_file', apply_trans, 'input_image')

        pipeline.connect_output('qsm_in_mni', apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline    
'''
   
''' Deprecated    
def qsm_se_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Process single echo data (TE=20ms)

        NB: Default values come from the STI-Suite
        """
        pipeline = self.create_pipeline(
            name='qsmrecon',
            inputs=[DatasetSpec('coils', directory_format)],
            # TODO should this be primary?
            outputs=[DatasetSpec('qsm_se', nifti_gz_format),
                     DatasetSpec('tissue_phase_se', nifti_gz_format),
                     DatasetSpec('tissue_mask_se', nifti_gz_format),
                     DatasetSpec('qsm_mask_se', nifti_gz_format)],
            description="Resolve QSM from t2star coils",
            default_options={},
            citations=[sti_cites, fsl_cite, matlab_cite],
            version=1,
            options=options)

        # Prepare and reformat SWI_COILS
        prepare = pipeline.create_node(interface=Prepare(), name='prepare',
                                       requirements=[matlab2015_req],
                                       wall_time=30, memory=8000)

        # Brain Mask
        mask = pipeline.create_node(interface=fsl.BET(), name='bet',
                                    requirements=[fsl5_req],
                                    wall_time=30, memory=8000)
        mask.inputs.reduce_bias = True
        mask.inputs.output_type = 'NIFTI_GZ'
        mask.inputs.frac = 0.3
        mask.inputs.mask = True

        # Phase and QSM for single echo
        qsmrecon = pipeline.create_node(interface=STI_SE(), name='qsmrecon',
                                        requirements=[matlab2015_req],
                                        wall_time=600, memory=16000)

        # Connect inputs/outputs
        pipeline.connect_input('coils', prepare, 'in_dir')
        pipeline.connect_output('qsm_mask_se', mask, 'mask_file_se')
        pipeline.connect_output('qsm_se', qsmrecon, 'qsm_se')
        pipeline.connect_output('tissue_phase_se', qsmrecon, 'tissue_phase_se')
        pipeline.connect_output('tissue_mask_se', qsmrecon, 'tissue_mask_se')

        pipeline.connect(prepare, 'out_file', mask, 'in_file')
        pipeline.connect(mask, 'mask_file_se', qsmrecon, 'mask_file_se')
        pipeline.connect(prepare, 'out_dir', qsmrecon, 'in_dir')

        pipeline.assert_connected()
        return pipeline
'''
    
''' Deprecated      
    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('t2s', nifti_gz_format),
        DatasetSpec('coils', directory_format,
                    description=("Reconstructed T2* complex image for each "
                                 "coil")),
        DatasetSpec('qsm', nifti_gz_format, qsm_pipeline,
                    description=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        DatasetSpec('tissue_phase', nifti_gz_format, qsm_pipeline,
                    description=("Phase map for each coil following unwrapping"
                                 " and background field removal")),
        DatasetSpec('tissue_mask', nifti_gz_format, qsm_pipeline,
                    description=("Mask for each coil corresponding to areas of"
                                 " high magnitude")),
        DatasetSpec('qsm_mask', nifti_gz_format, qsm_pipeline,
                    description=("Brain mask generated from T2* image")),
        DatasetSpec('betted_file', nifti_gz_format, optiBET),
        DatasetSpec('betted_mask', nifti_gz_format, optiBET),
        DatasetSpec('t2s_mask', nifti_gz_format, optiBET_T2s),
        DatasetSpec('T2s2T1', nifti_gz_format, ANTsRegistration),
        DatasetSpec('T2s2T1_mat', text_matrix_format, ANTsRegistration),
        DatasetSpec('T12MNI_linear', nifti_gz_format, ANTsRegistration),
        DatasetSpec('T12MNI_mat', text_matrix_format, ANTsRegistration),
        DatasetSpec('T12MNI_warp', nifti_gz_format, ANTsRegistration),
        DatasetSpec('T12MNI_invwarp', nifti_gz_format, ANTsRegistration),
        DatasetSpec('qsm_in_mni', nifti_gz_format, applyTransform),
        
        DatasetSpec('qsm_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        DatasetSpec('tissue_phase_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Phase map for each coil following unwrapping"
                                 " and background field removal")),
        DatasetSpec('tissue_mask_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Mask for each coil corresponding to areas of"
                                 " high magnitude")),
        DatasetSpec('qsm_mask_se', nifti_gz_format, qsm_se_pipeline,
                    description=("Brain mask generated from T2* image"))
        )
           
        DatasetSpec('melodic_dir', zip_format, feat_pipeline),
        DatasetSpec('train_data', rdata_format, TrainingFix),
        DatasetSpec('filtered_data', nifti_gz_format, rsfMRI_filtering),
        DatasetSpec('hires2example', text_matrix_format, rsfMRI_filtering),
        DatasetSpec('mc_par', par_format, rsfMRI_filtering),
        DatasetSpec('rsfmri_mask', nifti_gz_format, rsfMRI_filtering),
        DatasetSpec('unwarped_file', nifti_gz_format, rsfMRI_filtering),
        DatasetSpec('melodic_ica', zip_format, MelodicL1),
        DatasetSpec('fix_dir', zip_format, PrepareFix),
        DatasetSpec('smoothed_file', nifti_gz_format, applySmooth))
'''