from .base import MriStudy
from nipype.interfaces.utility import Split
from arcana import ParamSpec, SwitchSpec, FilesetSpec, StudyMetaClass
from banana.exceptions import BananaUsageError
from nipype.interfaces import fsl
from banana.requirement import fsl_req
from banana.citation import fsl_cite
from banana.file_format import nifti_gz_format, nifti_gz_x_format
from banana.bids_ import BidsInputs


class T2Study(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('wm_seg', nifti_gz_format, 'segmentation_pipeline')]

    add_param_specs = [
        SwitchSpec('bet_robust', True),
        ParamSpec('bet_f_threshold', 0.5),
        ParamSpec('bet_reduce_bias', False)]

    default_bids_inputs = [
        BidsInputs(spec_name='magnitude', type='T2w',
                   valid_formats=(nifti_gz_x_format, nifti_gz_format))]

    def segmentation_pipeline(self, img_type=2, **name_maps):

        pipeline = self.new_pipeline(
            name='FAST_segmentation',
            name_maps=name_maps,
            desc="White matter segmentation of the reference image",
            citations=[fsl_cite])

        fast = pipeline.add(
            'fast',
            fsl.FAST(
                img_type=img_type,
                segments=True,
                out_basename='Reference_segmentation',
                output_type='NIFTI_GZ'),
            inputs={
                'in_files': ('brain', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        # Determine output field of split to use
        if img_type == 1:
            split_output = 'out3'
        elif img_type == 2:
            split_output = 'out2'
        else:
            raise BananaUsageError(
                "'img_type' parameter can either be 1 or 2 (not {})"
                .format(img_type))

        pipeline.add(
            'split',
            Split(
                splits=[1, 1, 1],
                squeeze=True),
            inputs={
                'inlist': (fast, 'tissue_class_files')},
            outputs={
                'wm_seg': (split_output, nifti_gz_format)})

        return pipeline
