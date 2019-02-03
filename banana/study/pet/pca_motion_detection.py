from arcana.study.base import StudyMetaClass
from arcana.data import FilesetSpec, FieldSpec
from banana.file_format import (list_mode_format, directory_format)
from banana.study.pet.base import PetStudy
from banana.interfaces.custom.pet import (
    PrepareUnlistingInputs, PETListModeUnlisting, SSRB, MergeUnlistingOutputs)
from banana.requirement import stir_req


class PETPCAMotionDetectionStudy(PetStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('list_mode', list_mode_format),
        FieldSpec('time_offset', int),
        FieldSpec('temporal_length', float),
        FieldSpec('num_frames', int),
        FilesetSpec('ssrb_sinograms', directory_format,
                    'sinogram_unlisting_pipeline')]

    def sinogram_unlisting_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('list_mode', list_mode_format),
#                     FieldSpec('time_offset', int),
#                     FieldSpec('temporal_length', float),
#                     FieldSpec('num_frames', int)],
#             outputs=[FilesetSpec('ssrb_sinograms', directory_format)],


        pipeline = self.new_pipeline(
            name='prepare_sinogram',
            desc=('Unlist pet listmode data into several sinograms and '
                         'perform ssrb compression to prepare data for motion '
                         'detection using PCA pipeline.'),
            references=[],
            **kwargs)

        prepare_inputs = pipeline.add(
            'prepare_inputs',
            PrepareUnlistingInputs())
        pipeline.connect_input('list_mode', prepare_inputs, 'list_mode')
        pipeline.connect_input('time_offset', prepare_inputs, 'time_offset')
        pipeline.connect_input('num_frames', prepare_inputs, 'num_frames')
        pipeline.connect_input('temporal_length', prepare_inputs,
                               'temporal_len')
        unlisting = pipeline.add(
            'unlisting',
            PETListModeUnlisting(), iterfield=['list_inputs'])
        pipeline.connect(prepare_inputs, 'out', unlisting, 'list_inputs')

        ssrb = pipeline.add(
            'ssrb',
            SSRB(),
            requirements=[stir_req.v('3.0')])
        pipeline.connect(unlisting, 'pet_sinogram', ssrb, 'unlisted_sinogram')

        merge = pipeline.add(
            'merge_sinograms',
            MergeUnlistingOutputs(),
            joinsource='unlisting',
            joinfield=['sinograms'])
        pipeline.connect(ssrb, 'ssrb_sinograms', merge, 'sinograms')
        pipeline.connect_output('ssrb_sinograms', merge, 'sinogram_folder')

        return pipeline
