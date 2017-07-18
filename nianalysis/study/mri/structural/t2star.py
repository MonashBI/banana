from nianalysis.requirements import fsl5_req, matlab2015_req, ants19_req
from nianalysis.citations import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.data_formats import directory_format, nifti_gz_format, text_matrix_format
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.qsm import STI, STI_SE, Prepare, FillHoles
from nianalysis.interfaces import utils
from ..base import MRIStudy
from nipype.interfaces import fsl, ants
from nianalysis.interfaces.ants import AntsRegSyn
import os
import subprocess as sp
from nianalysis import pipeline

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
            inputs=[DatasetSpec('coils', directory_format)],
            # TODO should this be primary?
            outputs=[DatasetSpec('qsm', nifti_gz_format),
                     DatasetSpec('tissue_phase', nifti_gz_format),
                     DatasetSpec('tissue_mask', nifti_gz_format),
                     DatasetSpec('qsm_mask', nifti_gz_format)],
            description="Resolve QSM from t2star coils",
            default_options={},
            citations=[sti_cites, fsl_cite, matlab_cite],
            version=1,
            options=options)

        # Prepare and reformat SWI_COILS
        prepare = pipeline.create_node(interface=Prepare(), name='prepare',
                                       requirements=[matlab2015_req],
                                       wall_time=30, memory=16000)

        # Brain Mask
        mask = pipeline.create_node(interface=fsl.BET(), name='bet',
                                    requirements=[fsl5_req],
                                    wall_time=30, memory=8000)
        mask.inputs.reduce_bias = True
        mask.inputs.output_type = 'NIFTI_GZ'
        mask.inputs.frac = 0.3
        mask.inputs.mask = True

        # Phase and QSM for dual echo
        qsmrecon = pipeline.create_node(interface=STI(), name='qsmrecon',
                                        requirements=[matlab2015_req],
                                        wall_time=600, memory=24000)

        # Connect inputs/outputs
        pipeline.connect_input('coils', prepare, 'in_dir')
        pipeline.connect_output('qsm_mask', mask, 'mask_file')
        pipeline.connect_output('qsm', qsmrecon, 'qsm')
        pipeline.connect_output('tissue_phase', qsmrecon, 'tissue_phase')
        pipeline.connect_output('tissue_mask', qsmrecon, 'tissue_mask')

        pipeline.connect(prepare, 'out_file', mask, 'in_file')
        pipeline.connect(mask, 'mask_file', qsmrecon, 'mask_file')
        pipeline.connect(prepare, 'out_dir', qsmrecon, 'in_dir')

        pipeline.assert_connected()
        return pipeline

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
    
    def linearT1toMNI(self, **options):
      
        pipeline = self.create_pipeline(
            name='ANTS_Reg_T1_to_MNI_Mat',
            inputs=[DatasetSpec('betted_T1', nifti_gz_format)],
            outputs=[DatasetSpec('T1_to_MNI_initial_mat', text_matrix_format),
                     DatasetSpec('T1_to_MNI_initial', nifti_gz_format)],
            description=("python implementation of Affine ANTS Reg for T1 to MNI"),           
            default_options={'MNI_template_T1': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[ants19_req],
            options=options)
                
        t1reg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='a',
                       out_prefix='T1_to_MNI'), name='ANTsReg', requirements=[ants19_req], memory=16000, wall_time=30)
        t1reg.input.ref_file = pipeline.options('MNI_template_T1')
        
        pipeline.connect_input('betted_T1', t1reg, 'input_file')
        pipeline.connect_output('T1_to_MNI_initial_mat', t1reg, 'regmat')
        pipeline.connect_output('T1_to_MNI_initial', t1reg, 'reg_file')
        
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

        cp_geom = pipeline.create_node(fsl.CopyGeom(), name='copy_geomery', requirements=[fsl5_req], memory=8000, wall_time=5)
        pipeline.connect_input('qsm', cp_geom, 'dest_file')
        pipeline.connect_input('t2s', cp_geom, 'in_file')
        
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
        pipeline.connect(cp_geom, 'out_file', apply_trans, 'input_image')
        pipeline.connect_output('qsm_in_mni', apply_trans, 'output_image')

        pipeline.assert_connected()
        return pipeline    
        
    def mniInT2s(self, **options):
        
        pipeline = self.create_pipeline(
            name='ANTsApplyTransform',
            inputs=[DatasetSpec('t2s', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_warp', nifti_gz_format),
                    DatasetSpec('T1_to_MNI_mat', text_matrix_format),
                    DatasetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[DatasetSpec('qsm_in_mni', nifti_gz_format)],
            description=("Transform data from T2s to MNI space"),
            default_options={'MNI_template': os.environ['FSLDIR']+'/data/'
                             'standard/MNI152_T1_2mm_brain.nii.gz'},
            version=1,
            citations=[ants19_req],
            options=options)

        merge_trans = pipeline.create_node(utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')
        pipeline.connect_input('T1_to_MNI_mat', merge_trans, 'in2')
        pipeline.connect_input('T1_to_MNI_warp', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform', requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans.inputs.input_image = pipeline.option('MNI_template')
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, True, True]
        
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
        
        return pipeline
    
    def red_nuclei_masks(self, **options):
        
        return pipeline
    
    def subcortical_structure_masks(self, **options):
        
        return pipeline
    
    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('t2s', nifti_gz_format),
        DatasetSpec('coils', directory_format,
                    description=("Reconstructed T2* complex image for each "
                                 "coil")),
                                           
        DatasetSpec('betted_T1', nifti_gz_format, bet_T1), 
        DatasetSpec('betted_T1_mask', nifti_gz_format, bet_T1),   
             
        DatasetSpec('betted_T2s', nifti_gz_format, bet_T2s),     
        DatasetSpec('betted_T2s_mask', nifti_gz_format, bet_T2s),
        
        DatasetSpec('opti_betted_T1', nifti_gz_format, optiBET_T1),
        DatasetSpec('opti_betted_T1_mask', nifti_gz_format, optiBET_T1),
        
        DatasetSpec('opti_betted_T2s', nifti_gz_format, optiBET_T2s),
        DatasetSpec('opti_betted_T2s_mask', nifti_gz_format, optiBET_T2s),
        
        DatasetSpec('T1_to_MNI_mat', text_matrix_format, nonLinearT1ToMNI),
        DatasetSpec('T2s_to_T1_mat', text_matrix_format, linearT2sToT1),
        DatasetSpec('T1_to_MNI_warp', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('MNI_to_T1_warp', nifti_gz_format, nonLinearT1ToMNI),
        
        DatasetSpec('T1_in_MNI', nifti_gz_format, nonLinearT1ToMNI),
        DatasetSpec('T2s_in_T1', nifti_gz_format, linearT2sToT1),
                                
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
    
#    _dataset_specs = set_dataset_specs(
#        DatasetSpec('t1', nifti_gz_format),
#        DatasetSpec('t2s', nifti_gz_format),
#        DatasetSpec('coils', directory_format,
#                    description=("Reconstructed T2* complex image for each "
#                                 "coil")),
#        DatasetSpec('qsm', nifti_gz_format, qsm_pipeline,
#                    description=("Quantitative susceptibility image resolved "
#                                 "from T2* coil images")),
#        DatasetSpec('tissue_phase', nifti_gz_format, qsm_pipeline,
#                    description=("Phase map for each coil following unwrapping"
#                                 " and background field removal")),
#        DatasetSpec('tissue_mask', nifti_gz_format, qsm_pipeline,
#                    description=("Mask for each coil corresponding to areas of"
#                                 " high magnitude")),
#        DatasetSpec('qsm_mask', nifti_gz_format, qsm_pipeline,
#                    description=("Brain mask generated from T2* image")),
#        DatasetSpec('betted_file', nifti_gz_format, optiBET),
#        DatasetSpec('betted_mask', nifti_gz_format, optiBET),
#        DatasetSpec('t2s_mask', nifti_gz_format, optiBET_T2s),
#        DatasetSpec('T2s2T1', nifti_gz_format, ANTsRegistration),
#        DatasetSpec('T2s2T1_mat', text_matrix_format, ANTsRegistration),
#        DatasetSpec('T12MNI_linear', nifti_gz_format, ANTsRegistration),
#        DatasetSpec('T12MNI_mat', text_matrix_format, ANTsRegistration),
#        DatasetSpec('T12MNI_warp', nifti_gz_format, ANTsRegistration),
#        DatasetSpec('T12MNI_invwarp', nifti_gz_format, ANTsRegistration),
#        DatasetSpec('qsm_in_mni', nifti_gz_format, applyTransform),
#        
#        DatasetSpec('qsm_se', nifti_gz_format, qsm_se_pipeline,
#                    description=("Quantitative susceptibility image resolved "
#                                 "from T2* coil images")),
#        DatasetSpec('tissue_phase_se', nifti_gz_format, qsm_se_pipeline,
#                    description=("Phase map for each coil following unwrapping"
#                                 " and background field removal")),
#        DatasetSpec('tissue_mask_se', nifti_gz_format, qsm_se_pipeline,
#                    description=("Mask for each coil corresponding to areas of"
#                                 " high magnitude")),
#        DatasetSpec('qsm_mask_se', nifti_gz_format, qsm_se_pipeline,
#                    description=("Brain mask generated from T2* image"))
#        )
            
#        DatasetSpec('melodic_dir', zip_format, feat_pipeline),
#        DatasetSpec('train_data', rdata_format, TrainingFix),
#        DatasetSpec('filtered_data', nifti_gz_format, rsfMRI_filtering),
#        DatasetSpec('hires2example', text_matrix_format, rsfMRI_filtering),
#        DatasetSpec('mc_par', par_format, rsfMRI_filtering),
#        DatasetSpec('rsfmri_mask', nifti_gz_format, rsfMRI_filtering),
#        DatasetSpec('unwarped_file', nifti_gz_format, rsfMRI_filtering),
#        DatasetSpec('melodic_ica', zip_format, MelodicL1),
#        DatasetSpec('fix_dir', zip_format, PrepareFix),
#        DatasetSpec('smoothed_file', nifti_gz_format, applySmooth))