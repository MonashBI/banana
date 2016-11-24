from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import BET
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import (
    nifti_gz_format)
from ..base import set_dataset_specs, Study
from nianalysis.dataset import DatasetSpec

#
# class BETInputSpec(FSLCommandInputSpec):
#     # We use position args here as list indices - so a negative number
#     # will put something on the end
#     in_file = File(exists=True,
#                    desc='input file to skull strip',
#                    argstr='%s', position=0, mandatory=True)
#     out_file = File(desc='name of output skull stripped image',
#                     argstr='%s', position=1, genfile=True, hash_files=False)
#     outline = traits.Bool(desc='create surface outline image',
#                           argstr='-o')
#     mask = traits.Bool(desc='create binary mask image',
#                        argstr='-m')
#     skull = traits.Bool(desc='create skull image',
#                         argstr='-s')
#     no_output = traits.Bool(argstr='-n',
#                             desc="Don't generate segmented output")
#     frac = traits.Float(desc='fractional intensity threshold',
#                         argstr='-f %.2f')
#     vertical_gradient = traits.Float(
#         argstr='-g %.2f',
#         desc='vertical gradient in fractional intensity threshold (-1, 1)')
#     radius = traits.Int(argstr='-r %d', units='mm',
#                         desc="head radius")
#     center = traits.List(traits.Int, desc='center of gravity in voxels',
#                          argstr='-c %s', minlen=0, maxlen=3,
#                          units='voxels')
#     threshold = traits.Bool(
#         argstr='-t',
#         desc="apply thresholding to segmented brain image and mask")
#     mesh = traits.Bool(argstr='-e',
#                        desc="generate a vtk mesh brain surface")
#     # the remaining 'options' are more like modes (mutually exclusive) that
#     # FSL actually implements in a shell script wrapper around the bet binary.
#     # for some combinations of them in specific order a call would not fail,
#     # but in general using more than one of the following is clearly not
#     # supported
#     _xor_inputs = ('functional', 'reduce_bias', 'robust', 'padding',
#                    'remove_eyes', 'surfaces', 't2_guided')
#     robust = traits.Bool(
#         desc='robust brain centre estimation (iterates BET several times)',
#         argstr='-R', xor=_xor_inputs)
#     padding = traits.Bool(
#         desc=('improve BET if FOV is very small in Z (by temporarily padding '
#               'end slices)'),
#         argstr='-Z', xor=_xor_inputs)
#     remove_eyes = traits.Bool(
#         desc='eye & optic nerve cleanup (can be useful in SIENA)',
#         argstr='-S', xor=_xor_inputs)
#     surfaces = traits.Bool(
#         desc=('run bet2 and then betsurf to get additional skull and scalp '
#               'surfaces (includes registrations)'),
#         argstr='-A', xor=_xor_inputs)
#     t2_guided = File(desc='as with creating surfaces, when also feeding in '
#                           'non-brain-extracted T2 (includes registrations)',
#                      argstr='-A2 %s', xor=_xor_inputs)
#     functional = traits.Bool(argstr='-F', xor=_xor_inputs,
#                              desc="apply to 4D fMRI data")
#     reduce_bias = traits.Bool(argstr='-B', xor=_xor_inputs,
#                               desc="bias field and neck cleanup")
#
#
# class BETOutputSpec(TraitedSpec):
#     out_file = File(
#         desc="path/name of skullstripped file (if generated)")
#     mask_file = File(
#         desc="path/name of binary brain mask (if generated)")
#     outline_file = File(
#         desc="path/name of outline file (if generated)")
#     meshfile = File(
#         desc="path/name of vtk mesh file (if generated)")
#     inskull_mask_file = File(
#         desc="path/name of inskull mask (if generated)")
#     inskull_mesh_file = File(
#         desc="path/name of inskull mesh outline (if generated)")
#     outskull_mask_file = File(
#         desc="path/name of outskull mask (if generated)")
#     outskull_mesh_file = File(
#         desc="path/name of outskull mesh outline (if generated)")
#     outskin_mask_file = File(
#         desc="path/name of outskin mask (if generated)")
#     outskin_mesh_file = File(
#         desc="path/name of outskin mesh outline (if generated)")
#     skull_mask_file = File(
#         desc="path/name of skull mask (if generated)")


class CoregisteredStudy(Study):

    def registration_pipeline(self):
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self._create_pipeline(
            name='segmentation',
            inputs=['reference', 'to_register'],
            outputs=['registered'],
            description="Registers a mr scan against a reference image",
            options={},
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=1)
        bet = pe.Node(interface=BET(), name='bet')
        # Connect inputs/outputs
        pipeline.connect_input('to_register', bet, '')
        pipeline.connect_output('registered', bet, '')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('registered', nifti_gz_format, registration_pipeline))
