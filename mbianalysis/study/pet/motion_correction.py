from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (nifti_gz_format, directory_format)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import StudyMetaClass
from mbianalysis.interfaces.custom.pet import (StaticMotionCorrection)
from nipype.interfaces.fsl import Merge, MCFLIRT, ImageMaths
from nianalysis.interfaces.utils import ListDir
from mbianalysis.study.pet.base import PETStudy
from nianalysis.requirements import fsl509_req


class FixedMAFMotionCorrection(PETStudy):

    add_default_options = {'maf_xmin': 100,
                           'maf_xsize': 130,
                           'maf_ymin': 100,
                           'maf_ysize': 130,
                           'maf_zmin': 20,
                           'maf_zsize': 100},

    def pet_data_preparation_pipeline(self, **kwargs):
        return (super(FixedMAFMotionCorrection, self).
                pet_data_preparation_pipeline(**kwargs))

    def pet_fov_cropping_pipeline(self, **kwargs):
        return (super(FixedMAFMotionCorrection, self).
                pet_fov_cropping_pipeline(**kwargs))

    def fixed_maf_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='fixed_maf',
            inputs=[DatasetSpec('pet_data_cropped', directory_format)],
            outputs=[DatasetSpec('fixed_maf_pet', nifti_gz_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will align all of them to a reference and "
                         "create a static PET image by averaging the realigned"
                         "images."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        list_dir = pipeline.create_node(ListDir(), name='list_pet_dir')
        pipeline.connect_input('pet_data_cropped', list_dir, 'directory')
        merge = pipeline.create_node(Merge(), name='pet_merge',
                                     requirements=[fsl509_req])
        pipeline.connect(list_dir, 'files', merge, 'in_files')
        merge.inputs.dimension = 't'

        mcflirt = pipeline.create_node(MCFLIRT(), name='mcflirt',
                                       requirements=[fsl509_req])
        mcflirt.inputs.cost = 'normmi'
        pipeline.connect(merge, 'merged_file', mcflirt, 'in_file')

        mean = pipeline.create_node(ImageMaths(), name='time_average',
                                    requirements=[fsl509_req])
        mean.inputs.op_string = '-Tmean'
        pipeline.connect(mcflirt, 'out_file', mean, 'in_file')

        pipeline.connect_output('fixed_maf_pet', mean, 'out_file')
        pipeline.assert_connected()
        return pipeline

    add_data_specs = [
        DatasetSpec('pet_dir', directory_format),
        DatasetSpec('fixed_maf_pet', nifti_gz_format, 'fixed_maf_pipeline')]


class StaticPETMotionCorrection(PETStudy):

    def pet_data_preparation_pipeline(self, **kwargs):
        return (super(FixedMAFMotionCorrection, self).
                pet_data_preparation_pipeline(**kwargs))

    def pet_fov_cropping_pipeline(self, **kwargs):
        return (super(FixedMAFMotionCorrection, self).
                pet_fov_cropping_pipeline(**kwargs))

    def static_motion_correction_pipeline_factory(self, StructAlignment=None,
                                                  **options):
        inputs = [DatasetSpec('pet_data_cropped', directory_format),
                  DatasetSpec('motion_detection_output', directory_format)]
        if StructAlignment is not None:
            inputs.append(DatasetSpec(StructAlignment, nifti_gz_format))

        pipeline = self.create_pipeline(
            name='static_mc',
            inputs=inputs,
            outputs=[DatasetSpec('static_pet_mc', directory_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will generate a motion corrected static PET"
                         "image using information extracted from the MR-based "
                         "motion detection pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        static_mc = pipeline.create_node(
            StaticMotionCorrection(), name='static_mc',
            requirements=[fsl509_req])
        pipeline.connect_input('pet_data_cropped', static_mc, 'pet_cropped')
        pipeline.connect_input('motion_detection_output', static_mc, 'md_dir')
        if StructAlignment is not None:
            pipeline.connect_input(StructAlignment, static_mc,
                                   'structural_image')
        pipeline.connect_output('static_pet_mc', static_mc, 'pet_mc_results')
        pipeline.assert_connected()
        return pipeline

    def static_motion_correction_pipeline(self, **kwargs):
        return self.static_motion_correction_pipeline_factory(
            StructAlignment=None)

    add_data_specs = [
        DatasetSpec('pet_dir', directory_format),
        DatasetSpec('motion_detection_output', directory_format),
        DatasetSpec('static_pet_mc', directory_format,
                    'static_motion_correction_pipeline')]
