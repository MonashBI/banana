from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, text_format,
    png_format, dicom_format)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_specs
from mbianalysis.interfaces.custom.pet import PETFovCropping, PreparePetDir
from nipype.interfaces.fsl import Merge, ExtractROI, MCFLIRT, ImageMaths
from nianalysis.interfaces.utils import SelectOne


class MotionCorrection():
    
    def fixed_maf_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='fixed_maf',
            inputs=[DatasetSpec('pet_dir', directory_format)],
            outputs=[DatasetSpec('fixed_maf_pet', nifti_gz_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will align all of them to a reference and "
                         "create a static PET image by averaging the realigned"
                         "images."),
            default_options={'xmin': 100, 'xsize': 130, 'ymin': 100, 'ysize':130,
                             'zmin': 20, 'zsize': 100},
            version=1,
            citations=[fsl_cite],
            options=options)
        
        prep_dir = pipeline.create_node(PreparePetDir(), name='prepare_pet')
        pipeline.connect_input('pet_dir', prep_dir, 'pet_dir')
        
        select = pipeline.create_node(SelectOne(), name='select_ref')
        pipeline.connect(prep_dir, 'pet_images', select, 'inlist')
        select.inputs.index = [0]
        crop_ref = pipeline.create_node(ExtractROI(), name='crop_ref')
        pipeline.connect(select, 'out', crop_ref, 'in_file')
        crop_ref.inputs.x_min = pipeline.options('xmin')
        crop_ref.inputs.x_size = pipeline.options('xsize')
        crop_ref.inputs.y_min = pipeline.options('ymin')
        crop_ref.inputs.y_size = pipeline.options('ysize')
        crop_ref.inputs.z_min = pipeline.options('zmin')
        crop_ref.inputs.z_size = pipeline.options('zsize')
        cropping = pipeline.create_map_node(PETFovCropping(), name='cropping',
                                            iterfield=['pet_image'])
        cropping.inputs.x_min = pipeline.options('xmin')
        cropping.inputs.x_size = pipeline.options('xsize')
        cropping.inputs.y_min = pipeline.options('ymin')
        cropping.inputs.y_size = pipeline.options('ysize')
        cropping.inputs.z_min = pipeline.options('zmin')
        cropping.inputs.z_size = pipeline.options('zsize')
        pipeline.connect(crop_ref, 'roi_file', cropping, 'ref_pet')
        pipeline.connect(prep_dir, 'pet_images', cropping, 'pet_image')
        
        merge = pipeline.create_node(Merge(), name='pet_merge')
        pipeline.connect(cropping, 'pet_cropped', merge, 'in_files')
        merge.inputs.dimension = 't'
        
        mcflirt = pipeline.create_node(MCFLIRT(), name='mcflirt')
        mcflirt.inputs.cost = 'normmi'
        pipeline.connect(merge, 'merged_file', mcflirt, 'in_file')
        
        mean = pipeline.create_node(ImageMaths(), name='time_average')
        mean.inputs.op_string = 'Tmean'
        pipeline.connect(mcflirt, 'out_file', mean, 'in_file')
        
        pipeline.connect_output('fixed_maf_pet', mean, 'out_file')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_specs(
        DatasetSpec('pet_dir', directory_format),
        DatasetSpec('fixed_maf_pet', nifti_gz_format, fixed_maf_pipeline))
        