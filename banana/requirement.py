import os
import os.path as op
from arcana.environment.requirement import (
    CliRequirement, MatlabPackageRequirement,
    matlab_req)  # @UnusedImport

# Command line requirements

mrtrix_req = CliRequirement('mrtrix', test_cmd='mrinfo')
ants_req = CliRequirement('ants', test_cmd='antsRegistration')
dcm2niix_req = CliRequirement('dcm2niix', test_cmd='dcm2niix')
freesurfer_req = CliRequirement('freesurfer', test_cmd='recon-all')
fix_req = CliRequirement('fix', test_cmd='fix')
afni_req = CliRequirement('afni', test_cmd='afni')
stir_req = CliRequirement('stir', test_cmd='stir_math', version_switch=None)
c3d_req = CliRequirement('c3d', test_cmd='c3d', version_switch=None)

# Matlab package requirements

spm_req = MatlabPackageRequirement('spm', test_func=None)
noddi_req = MatlabPackageRequirement('noddi'),
niftimatlab_req = MatlabPackageRequirement('niftimatlib')
sti_req = MatlabPackageRequirement('sti')


class FSLRequirement(CliRequirement):

    def __init__(self):
        super().__init__('fsl', None)

    def detect_version(self):
        """
        As FSL doesn't have a simple way of printing the version, the best
        we can do is to read the copyright statement of a shell command.

        Note that this doesn't pick up the micro release
        """
        with open(op.join(os.getenv('FSLDIR'), 'etc', 'fslversion'), 'r') as f:
            contents = f.read()
        return self.parse_version(contents.strip())


fsl_req = FSLRequirement()

# mrtrix0_3_req = CliRequirement('mrtrix', min_version=(0, 3, 12),
#                                max_version=(0, 3, 15), test_cmd='mrconvert')
# mrtrix3_req = CliRequirement('mrtrix', min_version=(3, 0, 0),
#                              test_cmd='mrconvert')
# fsl5_req = CliRequirement('fsl', min_version=(5, 0, 8), test_cmd='fsl')
# fsl509_req = CliRequirement('fsl', min_version=(5, 0, 9),
#                             max_version=(5, 0, 9), test_cmd='fsl')
# fsl510_req = CliRequirement('fsl', min_version=(5, 0, 10),
#                             max_version=(5, 0, 10), test_cmd='fsl')
# ants2_req = CliRequirement('ants', min_version=(2, 0),
#                            test_cmd='antsRegistration')
# ants19_req = CliRequirement('ants', min_version=(1, 9),
#                             test_cmd='antsRegistration')
# spm12_req = MatlabRequirement('spm', min_version=(12, 0),
#                               test_func=None)
# freesurfer_req = CliRequirement('freesurfer', min_version=(5, 3))
# matlab2014_req = CliRequirement('matlab', min_version=(2014, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')
# matlab2015_req = CliRequirement('matlab', min_version=(2015, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')
# noddi_req = MatlabRequirement('noddi', min_version=(0, 9)),
# niftimatlab_req = MatlabRequirement('niftimatlib', (1, 2))
# dcm2niix_req = CliRequirement('dcm2niix', min_version=(1, 0, 2),
#                               test_cmd='dcm2niix')
# fix_req = CliRequirement('fix', min_version=(1, 0))
# afni_req = CliRequirement('afni', min_version=(16, 2, 10))
# mricrogl_req = CliRequirement('mricrogl', min_version=(1, 0, 20170207))
# stir_req = CliRequirement('stir', min_version=(3, 0))
# c3d_req = CliRequirement('c3d', min_version=(1, 1, 0))

# matlab2015_req = Requirement('matlab', min_version=(2015, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')

if __name__ == '__main__':

    for req in (mrtrix_req):  #, fsl_req, ants_req, freesurfer_req, matlab_req, noddi_req, niftimatlab_req, dcm2niix_req, fix_req, afni_req, stir_req, c3d_req):
        print('{}: {}'.format(req.detect_version()))
