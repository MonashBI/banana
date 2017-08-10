import os.path
from nipype.interfaces.base import (
    traits, InputMultiPath, File, TraitedSpec, isdefined)
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)
from nipype.interfaces.mrtrix3.preprocess import (
    ResponseSD as NipypeResponseSD,
    ResponseSDInputSpec as NipypeResponseSDInputSpec)
from nianalysis.utils import split_extension

# fod2fixel
# fixel2voxel
# fixelcorrespondence
# fixelcfestats
# tcksift
# warp2metric
