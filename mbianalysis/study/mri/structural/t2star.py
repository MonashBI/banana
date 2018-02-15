from nianalysis.requirements import fsl5_req, matlab2015_req
from nianalysis.citations import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.data_formats import directory_format, nifti_gz_format
from nianalysis.study.base import set_data_specs
from nianalysis.dataset import DatasetSpec
from mbianalysis.interfaces.qsm import STI, STI_SE, Prepare
from ..base import MRIStudy
from nipype.interfaces import fsl


class T2StarStudy(MRIStudy):

    def qsm_de_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
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
        qsmrecon = pipeline.create_node(interface=STI_DE(), name='qsmrecon',
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
        
        bias = pipeline.create_node(interface=ants.N4BiasFieldCorrection(),
                                    name='n4_bias_correction', requirements=[ants19_req],
                                    wall_time=60, memory=12000)
        pipeline.connect_input('t1', bias, 'input_image')
        
        bet = pipeline.create_node(
            fsl.BET(frac=0.15, reduce_bias=True), name='bet', requirements=[fsl5_req], memory=8000, wall_time=45)
            
        pipeline.connect(bias,'output_image', bet, 'in_file')
        pipeline.connect_output('betted_T1', bet, 'out_file')
        pipeline.connect_output('betted_T1_mask', bet, 'mask_file')
        
        return pipeline
    
    def cet_T1(self, **options):
        pipeline = self.create_pipeline(
            name='CET_T1',
            inputs=[DatasetSpec('betted_T1', nifti_gz_format),
                    DatasetSpec(self._lookup_l_tfm_to_name('MNI'), text_matrix_format),
                    DatasetSpec(self._lookup_nl_tfm_inv_name('MNI'), nifti_gz_format)],
            outputs=[DatasetSpec('cetted_T1_mask', nifti_gz_format),
                     DatasetSpec('cetted_T1', nifti_gz_format)],
            description=("Construct cerebellum mask using SUIT template"),
            default_options={'SUIT_mask': self._lookup_template_mask_path('SUIT')},
            version=1,
            citations=[fsl_cite],
            options=options)
        
        # Initially use MNI space to warp SUIT into T1 and threshold to mask
        merge_trans = pipeline.create_node(utils.Merge(2), name='merge_transforms')
        pipeline.connect_input(self._lookup_nl_tfm_inv_name('MNI'), merge_trans, 'in2')
        pipeline.connect_input(self._lookup_l_tfm_to_name('MNI'), merge_trans, 'in1')


    def qsm_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Process single echo data (TE=20ms)

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
        qsmrecon = pipeline.create_node(interface=STI(), name='qsmrecon',
                                        requirements=[matlab2015_req],
                                        wall_time=600, memory=16000)

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

    _data_specs = set_data_specs(
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
                    description=("Brain mask generated from T2* image")))
