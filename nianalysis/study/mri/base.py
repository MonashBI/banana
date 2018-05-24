from nipype.interfaces import fsl
from nipype.interfaces.spm.preprocess import Coregister
from nianalysis.requirement import spm12_req
from nianalysis.citation import spm_cite
from nianalysis.data_format import nifti_format, motion_mats_format,\
    directory_format, nifti_gz_format
from arcana.dataset import DatasetSpec, FieldSpec
from arcana.study.base import Study, StudyMetaClass
from nianalysis.citation import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_format import (
    dicom_format, text_format, gif_format)
from nianalysis.requirement import (fsl5_req, mrtrix3_req, fsl509_req,
                                    ants2_req, dcm2niix_req)
from nipype.interfaces.fsl import (FLIRT, FNIRT, Reorient2Std)
from nianalysis.utils import get_atlas_path
from arcana.exception import (
    ArcanaError, ArcanaUsageError)
from nianalysis.interfaces.mrtrix.transform import MRResize
from nianalysis.interfaces.custom.dicom import (DicomHeaderInfoExtraction)
from nipype.interfaces.utility import Split, Merge
from nianalysis.interfaces.fsl import FSLSlices
from nianalysis.data_format import text_matrix_format
import os
import logging
from nianalysis.interfaces.ants import AntsRegSyn
from nipype.interfaces.ants.resampling import ApplyTransforms
from arcana.option import OptionSpec
from nianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)

logger = logging.getLogger('Arcana')

atlas_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'atlases'))


class MRIStudy(Study):

    BRAIN_MASK_NAME = 'brain_mask'
    COREGISTER_TO_ATLAS_NAME = 'coregister_to_atlas'

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('primary', dicom_format),
        DatasetSpec('coreg_ref_brain', nifti_gz_format,
                    desc=("A reference scan to coregister the primary "
                          "scan to. Should be brain extracted"),
                    optional=True),
        DatasetSpec('coreg_matrix', text_matrix_format,
                    'linear_coregistration_pipeline'),
        DatasetSpec('preproc', nifti_gz_format,
                    'basic_preproc_pipeline'),
        DatasetSpec('brain', nifti_gz_format, 'brain_mask_pipeline',
                    desc="The brain masked image"),
        DatasetSpec('brain_mask', nifti_gz_format,
                    'brain_mask_pipeline',
                    desc="Mask of the brain"),
        DatasetSpec('coreg_brain', nifti_gz_format,
                    'linear_coregistration_pipeline',
                    desc=""),
        DatasetSpec('coreg_to_atlas', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        DatasetSpec('wm_seg', nifti_gz_format,
                    'segmentation_pipeline'),
        DatasetSpec('dcm_info', text_format,
                    'header_info_extraction_pipeline'),
        DatasetSpec('motion_mats', motion_mats_format,
                    'motion_mat_pipeline'),
        DatasetSpec('qformed', nifti_gz_format,
                    'qform_transform_pipeline'),
        DatasetSpec('qform_mat', text_matrix_format,
                    'qform_transform_pipeline'),
        FieldSpec('tr', float, 'header_info_extraction_pipeline'),
        FieldSpec('start_time', str, 'header_info_extraction_pipeline'),
        FieldSpec('real_duration', str, 'header_info_extraction_pipeline'),
        FieldSpec('tot_duration', str, 'header_info_extraction_pipeline'),
        FieldSpec('ped', str, 'header_info_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_info_extraction_pipeline')]

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.5),
        OptionSpec('bet_reduce_bias', False),
        OptionSpec('bet_g_threshold', 0.0),
        OptionSpec('bet_method', 'fsl_bet',
                   choices=('fsl_bet', 'optibet')),
        OptionSpec('MNI_template',
                   os.path.join(atlas_path, 'MNI152_T1_2mm.nii.gz')),
        OptionSpec('MNI_template_mask', os.path.join(
            atlas_path, 'MNI152_T1_2mm_brain_mask.nii.gz')),
        OptionSpec('optibet_gen_report', False),
        OptionSpec('fnirt_atlas_reg_tool', 'fnirt'),
        OptionSpec('fnirt_atlas', 'MNI152'),
        OptionSpec('fnirt_resolution', '2mm'),
        OptionSpec('fnirt_intensity_model', 'global_non_linear_with_bias'),
        OptionSpec('fnirt_subsampling', [4, 4, 2, 2, 1, 1]),
        OptionSpec('preproc_new_dims', ('RL', 'AP', 'IS')),
        OptionSpec('preproc_resolution', None, dtype=list),
        OptionSpec('linear_reg_method', 'flirt',
                   choices=('flirt', 'spm')),
        OptionSpec('flirt_degrees_of_freedom', 6, desc=(
            "Number of degrees of freedom used in the registration. "
            "Default is 6 -> affine transformation.")),
        OptionSpec('flirt_cost_func', 'normmi', desc=(
            "Cost function used for the registration. Can be one of "
            "'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',"
            " 'labeldiff', 'bbr'")),
        OptionSpec('flirt_qsform', False, desc=(
            "Whether to use the QS form supplied in the input image "
            "header (the image coordinates of the FOV supplied by the "
            "scanner"))]

    @property
    def coreg_brain_spec(self):
        """
        The name of the dataset after registration has been applied.
        If registration is not required, i.e. a reg_ref is not supplied
        then it is simply the 'brain' dataset.
        """
        if 'coreg_ref_brain' in self.input_names:
            name = 'coreg_brain'
        else:
            name = 'brain'
        return DatasetSpec(name, nifti_gz_format)

    def linear_coregistration_pipeline(self, **kwargs):
        pipeline_name = 'linear_coreg'
        method = self.pre_option('linear_reg_method', pipeline_name,
                                 **kwargs)
        if method == 'flirt':
            pipeline = self._flirt_factory(
                pipeline_name, 'brain', 'coreg_ref_brain',
                'coreg_brain', 'coreg_matrix', **kwargs)
        elif method == 'spm':
            raise NotImplementedError
        else:
            assert False
        return pipeline

    def qform_transform_pipeline(self, **kwargs):
        return self._qform_transform_factory(
            'qform_transform', 'brain', 'coreg_ref_brain', 'qformed',
            'qform_mat', **kwargs)

    def _flirt_factory(self, name, to_reg, ref, reg, matrix, **kwargs):
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command

        Parameters
        ----------
        name : str
            Name for the generated pipeline
        to_reg : str
            Name of the DatasetSpec to register
        ref : str
            Name of the DatasetSpec to use as a reference
        reg : str
            Name of the DatasetSpec to output as registered image
        matrix : str
            Name of the DatasetSpec to output as registration matrix
        """

        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(to_reg, nifti_gz_format),
                    DatasetSpec(ref, nifti_gz_format)],
            outputs=[DatasetSpec(reg, nifti_gz_format),
                     DatasetSpec(matrix, text_matrix_format)],
            desc="Registers a MR scan against a reference image",
            version=1,
            citations=[fsl_cite],
            **kwargs)
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req],
                                     wall_time=5)

        # Set registration options
        flirt.inputs.dof = pipeline.option('flirt_degrees_of_freedom')
        flirt.inputs.cost = pipeline.option('flirt_cost_func')
        flirt.inputs.cost_func = pipeline.option('flirt_cost_func')
        flirt.inputs.output_type = 'NIFTI_GZ'
        # Connect inputs
        pipeline.connect_input(to_reg, flirt, 'in_file')
        pipeline.connect_input(ref, flirt, 'reference')
        # Connect outputs
        pipeline.connect_output(reg, flirt, 'out_file')
        pipeline.connect_output(matrix, flirt,
                                'out_matrix_file')
        return pipeline

    def _qform_transform_factory(self, name, to_reg, ref, qformed,
                                 qformed_mat, **kwargs):
        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(to_reg, nifti_gz_format),
                    DatasetSpec(ref, nifti_gz_format)],
            outputs=[DatasetSpec(qformed, nifti_gz_format),
                     DatasetSpec(qformed_mat, text_matrix_format)],
            desc="Registers a MR scan against a reference image",
            version=1,
            citations=[fsl_cite],
            **kwargs)
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req],
                                     wall_time=5)
        flirt.inputs.uses_qform = True
        flirt.inputs.apply_xfm = True
        # Connect inputs
        pipeline.connect_input(to_reg, flirt, 'in_file')
        pipeline.connect_input(ref, flirt, 'reference')
        # Connect outputs
        pipeline.connect_output(qformed, flirt, 'out_file')
        pipeline.connect_output(qformed_mat, flirt, 'out_matrix_file')
        return pipeline

    def _spm_coreg_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Coregisters T2 image to T1 image using SPM's
        "Register" method.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='registration',
            inputs=[DatasetSpec('t1', nifti_format),
                    DatasetSpec('t2', nifti_format)],
            outputs=[DatasetSpec('t2_coreg_t1', nifti_format)],
            desc="Coregister T2-weighted images to T1",
            version=1,
            citations=[spm_cite],
            **kwargs)
        coreg = pipeline.create_node(Coregister(), name='coreg',
                                     requirements=[spm12_req], wall_time=30)
        coreg.inputs.jobtype = 'estwrite'
        coreg.inputs.cost_function = 'nmi'
        coreg.inputs.separation = [4, 2]
        coreg.inputs.tolerance = [
            0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001,
            0.001, 0.001]
        coreg.inputs.fwhm = [7, 7]
        coreg.inputs.write_interp = 4
        coreg.inputs.write_wrap = [0, 0, 0]
        coreg.inputs.write_mask = False
        coreg.inputs.out_prefix = 'r'
        # Connect inputs
        pipeline.connect_input('t1', coreg, 'target')
        pipeline.connect_input('t2', coreg, 'source')
        # Connect outputs
        pipeline.connect_output('t2_coreg_t1', coreg, 'coregistered_source')
        return pipeline

    def brain_mask_pipeline(self, in_file='preproc', **kwargs):
        bet_method = self.pre_option('bet_method', self.BRAIN_MASK_NAME,
                                     **kwargs)
        if bet_method == 'fsl_bet':
            pipeline = self._fsl_bet_brain_mask_pipeline(in_file, **kwargs)
        elif bet_method == 'optibet':
            pipeline = self._optiBET_brain_mask_pipeline(in_file, **kwargs)
        else:
            raise ArcanaError("Unrecognised brain extraction tool '{}'"
                              .format(bet_method))
        return pipeline

    def _fsl_bet_brain_mask_pipeline(self, in_file, name='', **kwargs):
        """
        Generates a whole brain mask using FSL's BET command.
        """
        pipeline = self.create_pipeline(
            name=name+self.BRAIN_MASK_NAME,
            inputs=[DatasetSpec(in_file, nifti_gz_format)],
            outputs=[DatasetSpec('brain', nifti_gz_format),
                     DatasetSpec('brain_mask', nifti_gz_format)],
            desc="Generate brain mask from mr_scan",
            version=1,
            citations=[fsl_cite, bet_cite, bet2_cite],
            **kwargs)
        # Create mask node
        bet = pipeline.create_node(interface=fsl.BET(), name="bet",
                                   requirements=[fsl509_req])
        bet.inputs.mask = True
        bet.inputs.output_type = 'NIFTI_GZ'
        if pipeline.option('bet_robust'):
            bet.inputs.robust = True
        if pipeline.option('bet_reduce_bias'):
            bet.inputs.reduce_bias = True
        bet.inputs.frac = pipeline.option('bet_f_threshold')
        bet.inputs.vertical_gradient = pipeline.option(
            'bet_g_threshold')
        # Connect inputs/outputs
        pipeline.connect_input(in_file, bet, 'in_file')
        pipeline.connect_output('brain', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        return pipeline

    def _optiBET_brain_mask_pipeline(self, in_file, name='', **kwargs):
        """
        Generates a whole brain mask using a modified optiBET approach.
        """

        outputs = [DatasetSpec('brain', nifti_gz_format),
                   DatasetSpec('brain_mask', nifti_gz_format)]
        if self.pre_option('optibet_gen_report', self.BRAIN_MASK_NAME,
                           **kwargs):
            outputs.append(DatasetSpec('optiBET_report', gif_format))
        pipeline = self.create_pipeline(
            name=name+self.BRAIN_MASK_NAME,
            inputs=[DatasetSpec(in_file, nifti_gz_format)],
            outputs=outputs,
            desc=("Modified implementation of optiBET.sh"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mni_reg = pipeline.create_node(
            AntsRegSyn(num_dimensions=3, transformation='s',
                       out_prefix='T12MNI', num_threads=1), name='T1_reg',
            wall_time=25, requirements=[ants2_req])
        mni_reg.inputs.ref_file = pipeline.option('MNI_template')
        pipeline.connect_input(in_file, mni_reg, 'input_file')

        merge_trans = pipeline.create_node(Merge(2), name='merge_transforms',
                                           wall_time=1)
        pipeline.connect(mni_reg, 'inv_warp', merge_trans, 'in1')
        pipeline.connect(mni_reg, 'regmat', merge_trans, 'in2')

        trans_flags = pipeline.create_node(Merge(2), name='trans_flags',
                                           wall_time=1)
        trans_flags.inputs.in1 = False
        trans_flags.inputs.in2 = True

        apply_trans = pipeline.create_node(
            ApplyTransforms(), name='ApplyTransform', wall_time=7,
            memory=24000, requirements=[ants2_req])
        apply_trans.inputs.input_image = pipeline.option('MNI_template_mask')
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect(trans_flags, 'out', apply_trans,
                         'invert_transform_flags')
        pipeline.connect_input(in_file, apply_trans, 'reference_image')

        maths1 = pipeline.create_node(
            fsl.ImageMaths(suffix='_optiBET_brain_mask', op_string='-bin'),
            name='binarize', wall_time=5, requirements=[fsl5_req])
        pipeline.connect(apply_trans, 'output_image', maths1, 'in_file')
        maths2 = pipeline.create_node(
            fsl.ImageMaths(suffix='_optiBET_brain', op_string='-mas'),
            name='mask', wall_time=5, requirements=[fsl5_req])
        pipeline.connect_input(in_file, maths2, 'in_file')
        pipeline.connect(maths1, 'out_file', maths2, 'in_file2')
        if pipeline.option('optibet_gen_report'):
            slices = pipeline.create_node(
                FSLSlices(), name='slices', wall_time=5,
                requirements=[fsl5_req])
            slices.inputs.outname = 'optiBET_report'
            pipeline.connect_input(in_file, slices, 'im1')
            pipeline.connect(maths2, 'out_file', slices, 'im2')
            pipeline.connect_output('optiBET_report', slices, 'report')

        pipeline.connect_output('brain_mask', maths1, 'out_file')
        pipeline.connect_output('brain', maths2, 'out_file')

        return pipeline

    def coregister_to_atlas_pipeline(self, **kwargs):
        atlas_reg_tool = self.option.get('atlas_reg_tool',
                                         self.COREGISTER_TO_ATLAS_NAME)
        if atlas_reg_tool == 'fnirt':
            pipeline = self._fsl_fnirt_to_atlas_pipeline(**kwargs)
        else:
            raise ArcanaError("Unrecognised coregistration tool '{}'"
                              .format(atlas_reg_tool))
        return pipeline

    # @UnusedVariable @IgnorePep8
    def _fsl_fnirt_to_atlas_pipeline(self, **kwargs):
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        atlas : Which atlas to use, can be one of 'mni_nl6'
        """
        pipeline = self.create_pipeline(
            name=self.COREGISTER_TO_ATLAS_NAME,
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('brain', nifti_gz_format)],
            outputs=[DatasetSpec('coreg_to_atlas', nifti_gz_format),
                     DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format)],
            desc=("Nonlinearly registers a MR scan to a standard space,"
                  "e.g. MNI-space"),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        # Get the reference atlas from FSL directory
        ref_atlas = get_atlas_path(pipeline.option('fnirt_atlas'), 'image',
                                   resolution=pipeline.option('resolution'))
        ref_mask = get_atlas_path(
            pipeline.option('fnirt_atlas'), 'mask_dilated',
            resolution=pipeline.option('resolution'))
        ref_brain = get_atlas_path(pipeline.option('fnirt_atlas'), 'brain',
                                   resolution=pipeline.option('resolution'))
        # Basic reorientation to standard MNI space
        reorient = pipeline.create_node(Reorient2Std(), name='reorient',
                                        requirements=[fsl5_req])
        reorient.inputs.output_type = 'NIFTI_GZ'
        reorient_mask = pipeline.create_node(
            Reorient2Std(), name='reorient_mask', requirements=[fsl5_req])
        reorient_mask.inputs.output_type = 'NIFTI_GZ'
        reorient_brain = pipeline.create_node(
            Reorient2Std(), name='reorient_brain', requirements=[fsl5_req])
        reorient_brain.inputs.output_type = 'NIFTI_GZ'
        # Affine transformation to MNI space
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req],
                                     wall_time=5)
        flirt.inputs.reference = ref_brain
        flirt.inputs.dof = 12
        flirt.inputs.output_type = 'NIFTI_GZ'
        # Nonlinear transformation to MNI space
        fnirt = pipeline.create_node(interface=FNIRT(), name='fnirt',
                                     requirements=[fsl5_req],
                                     wall_time=60)
        fnirt.inputs.ref_file = ref_atlas
        fnirt.inputs.refmask_file = ref_mask
        fnirt.inputs.output_type = 'NIFTI_GZ'
        intensity_model = pipeline.option('fnirt_intensity_model')
        if intensity_model is None:
            intensity_model = 'none'
        fnirt.inputs.intensity_mapping_model = intensity_model
        fnirt.inputs.subsampling_scheme = pipeline.option('fnirt_subsampling')
        fnirt.inputs.fieldcoeff_file = True
        fnirt.inputs.in_fwhm = [8, 6, 5, 4.5, 3, 2]
        fnirt.inputs.ref_fwhm = [8, 6, 5, 4, 2, 0]
        fnirt.inputs.regularization_lambda = [300, 150, 100, 50, 40, 30]
        fnirt.inputs.apply_intensity_mapping = [1, 1, 1, 1, 1, 0]
        fnirt.inputs.max_nonlin_iter = [5, 5, 5, 5, 5, 10]
        # Apply mask if corresponding subsampling scheme is 1
        # (i.e. 1-to-1 resolution) otherwise don't.
        apply_mask = [int(s == 1)
                      for s in pipeline.option('fnirt_subsampling')]
        fnirt.inputs.apply_inmask = apply_mask
        fnirt.inputs.apply_refmask = apply_mask
        # Connect nodes
        pipeline.connect(reorient_brain, 'out_file', flirt, 'in_file')
        pipeline.connect(reorient, 'out_file', fnirt, 'in_file')
        pipeline.connect(reorient_mask, 'out_file', fnirt, 'inmask_file')
        pipeline.connect(flirt, 'out_matrix_file', fnirt, 'affine_file')
        # Set registration options
        # TOD: Need to work out which options to use
        # Connect inputs
        pipeline.connect_input('preproc', reorient, 'in_file')
        pipeline.connect_input('brain_mask', reorient_mask, 'in_file')
        pipeline.connect_input('brain', reorient_brain, 'in_file')
        # Connect outputs
        pipeline.connect_output('coreg_to_atlas', fnirt, 'warped_file')
        pipeline.connect_output('coreg_to_atlas_coeff', fnirt,
                                'fieldcoeff_file')
        return pipeline

    def segmentation_pipeline(self, img_type=2, **kwargs):
        pipeline = self.create_pipeline(
            name='FAST_segmentation',
            inputs=[DatasetSpec('brain', nifti_gz_format)],
            outputs=[DatasetSpec('wm_seg', nifti_gz_format)],
            desc="White matter segmentation of the reference image",
            version=1,
            citations=[fsl_cite],
            **kwargs)

        fast = pipeline.create_node(fsl.FAST(), name='fast',
                                    requirements=[fsl509_req])
        fast.inputs.img_type = img_type
        fast.inputs.segments = True
        fast.inputs.out_basename = 'Reference_segmentation'
        pipeline.connect_input('brain', fast, 'in_files')
        split = pipeline.create_node(Split(), name='split')
        split.inputs.splits = [1, 1, 1]
        split.inputs.squeeze = True
        pipeline.connect(fast, 'tissue_class_files', split, 'inlist')
        if img_type == 1:
            pipeline.connect_output('wm_seg', split, 'out3')
        elif img_type == 2:
            pipeline.connect_output('wm_seg', split, 'out2')
        else:
            raise ArcanaUsageError(
                "'img_type' option can either be 1 or 2 (not {})"
                .format(img_type))

        return pipeline

    def basic_preproc_pipeline(self, in_file_name='primary', **kwargs):
        """
        Performs basic preprocessing, such as swapping dimensions into
        standard orientation and resampling (if required)

        Options
        -------
        new_dims : tuple(str)[3]
            A 3-tuple with the new orientation of the image (see FSL
            swap dim)
        resolution : list(float)[3] | None
            New resolution of the image. If None no resampling is
            performed
        """
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
#         swap.inputs.new_dims = pipeline.option('preproc_new_dims')
        pipeline.connect_input(in_file_name, swap, 'in_file')
        if pipeline.option('preproc_resolution') is not None:
            resample = pipeline.create_node(MRResize(), name="resample",
                                            requirements=[mrtrix3_req])
            resample.inputs.voxel = pipeline.option('preproc_resolution')
            pipeline.connect(swap, 'out_file', resample, 'in_file')
            pipeline.connect_output('preproc', resample, 'out_file')
        else:
            pipeline.connect_output('preproc', swap, 'out_file')

        return pipeline

    def header_info_extraction_pipeline(self, **kwargs):
        if self.input('primary').format != dicom_format:
            raise ArcanaUsageError(
                "Can only extract header info if 'primary' dataset "
                "is provided in DICOM format ({})".format(
                    self.input('primary').format))
        return self.header_info_extraction_pipeline_factory(
            'header_info_extraction', 'primary', **kwargs)

    def header_info_extraction_pipeline_factory(
            self, name, dcm_in_name, multivol=False, output_prefix='',
            **kwargs):

        tr = output_prefix + 'tr'
        start_time = output_prefix + 'start_time'
        tot_duration = output_prefix + 'tot_duration'
        real_duration = output_prefix + 'real_duration'
        ped = output_prefix + 'ped'
        pe_angle = output_prefix + 'pe_angle'
        dcm_info = output_prefix + 'dcm_info'
        outputs = [FieldSpec(tr, dtype=float),
                   FieldSpec(start_time, dtype=str),
                   FieldSpec(tot_duration, dtype=str),
                   FieldSpec(real_duration, dtype=str),
                   FieldSpec(ped, dtype=str),
                   FieldSpec(pe_angle, dtype=str),
                   DatasetSpec(dcm_info, text_format)]

        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(dcm_in_name, dicom_format)],
            outputs=outputs,
            desc=("Pipeline to extract the most important scan "
                  "information from the image header"),
            version=1,
            citations=[],
            **kwargs)
        hd_extraction = pipeline.create_node(DicomHeaderInfoExtraction(),
                                             name='hd_info_extraction')
        hd_extraction.inputs.multivol = multivol
        pipeline.connect_input(dcm_in_name, hd_extraction, 'dicom_folder')
        pipeline.connect_output(tr, hd_extraction, 'tr')
        pipeline.connect_output(start_time, hd_extraction, 'start_time')
        pipeline.connect_output(
            tot_duration, hd_extraction, 'tot_duration')
        pipeline.connect_output(
            real_duration, hd_extraction, 'real_duration')
        pipeline.connect_output(ped, hd_extraction, 'ped')
        pipeline.connect_output(pe_angle, hd_extraction, 'pe_angle')
        pipeline.connect_output(dcm_info, hd_extraction, 'dcm_info')
        return pipeline

    def motion_mat_pipeline(self, **kwargs):
        if not self.spec('coreg_matrix').derivable:
            logger.info("Cannot derive 'coreg_matrix' for {} required for "
                        "motion matrix calculation, assuming that it "
                        "is the reference study".format(self))
            inputs = [DatasetSpec('primary', dicom_format)]
            ref = True
        else:
            inputs = [DatasetSpec('coreg_matrix', text_matrix_format),
                      DatasetSpec('qform_mat', text_matrix_format)]
            if 'align_mats' in self.data_spec_names():
                inputs.append(DatasetSpec('align_mats', directory_format))
            ref = False
        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=inputs,
            outputs=[DatasetSpec('motion_mats', motion_mats_format)],
            desc=("Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        if ref:
            mm.inputs.reference = True
            pipeline.connect_input('primary', mm, 'dummy_input')
        else:
            pipeline.connect_input('coreg_matrix', mm, 'reg_mat')
            pipeline.connect_input('qform_mat', mm, 'qform_mat')
            if 'align_mats' in self.data_spec_names():
                pipeline.connect_input('align_mats', mm, 'align_mats')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline

