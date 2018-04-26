from nianalysis.study.base import StudyMetaClass
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (list_mode_format, directory_format)
from mbianalysis.study.pet.base import PETStudy
from mbianalysis.interfaces.custom.pet import (
    PrepareUnlistingInputs, PETListModeUnlisting, SSRB, MergeUnlistingOutputs)
from nianalysis.requirements import stir_req


class PETPCAMotionDetectionStudy(PETStudy):

    __metaclass__ = StudyMetaClass

    def sinogram_unlisting_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='prepare_sinogram',
            inputs=[DatasetSpec('list_mode', list_mode_format),
                    FieldSpec('time_offset', int),
                    FieldSpec('temporal_length', float),
                    FieldSpec('num_frames', int)],
            outputs=[DatasetSpec('ssrb_sinograms', directory_format)],
            description=('Unlist pet listmode data into several sinograms and '
                         'perform ssrb compression to prepare data for motion '
                         'detection using PCA pipeline.'),
            default_options={},
            version=1,
            citations=[],
            **kwargs)

        prepare_inputs = pipeline.create_node(PrepareUnlistingInputs(),
                                              name='prepare_inputs')
        pipeline.connect_input('list_mode', prepare_inputs, 'list_mode')
        pipeline.connect_input('time_offset', prepare_inputs, 'time_offset')
        pipeline.connect_input('num_frames', prepare_inputs, 'num_frames')
        pipeline.connect_input('temporal_length', prepare_inputs,
                               'temporal_len')
        unlisting = pipeline.create_node(
            PETListModeUnlisting(), iterfield=['list_inputs'],
            name='unlisting')
        pipeline.connect(prepare_inputs, 'out', unlisting, 'list_inputs')

        ssrb = pipeline.create_node(
            SSRB(), name='ssrb', requirements=[stir_req])
        pipeline.connect(unlisting, 'pet_sinogram', ssrb, 'unlisted_sinogram')

        merge = pipeline.create_join_node(
            MergeUnlistingOutputs(), joinsource='unlisting',
            joinfield=['sinograms'], name='merge_sinograms')
        pipeline.connect(ssrb, 'ssrb_sinograms', merge, 'sinograms')
        pipeline.connect_output('ssrb_sinograms', merge, 'sinogram_folder')

        pipeline.assert_connected()
        return pipeline

    add_data_specs = [
        DatasetSpec('list_mode', list_mode_format),
        FieldSpec('time_offset', int),
        FieldSpec('temporal_length', float),
        FieldSpec('num_frames', int),
        DatasetSpec('ssrb_sinograms', directory_format,
                    'sinogram_unlisting_pipeline')]
