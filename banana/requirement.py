import os
import os.path as op
import re
from arcana.environment.requirement import (
    CliRequirement, MatlabPackageRequirement, PythonPackageRequirement,
    matlab_req)  # @UnusedImport
from arcana.exception import (
    ArcanaRequirementNotFoundError, ArcanaRequirementVersionNotDectableError)

# Command line requirements

mrtrix_req = CliRequirement('mrtrix', test_cmd='mrinfo')
ants_req = CliRequirement('ants', test_cmd='antsRegistration')
dcm2niix_req = CliRequirement('dcm2niix', test_cmd='dcm2niix')
freesurfer_req = CliRequirement('freesurfer', test_cmd='recon-all')
fix_req = CliRequirement('fix', test_cmd='fix', version_switch='')
afni_req = CliRequirement('afni', test_cmd='afni')
stir_req = CliRequirement('stir', test_cmd='stir_math', version_switch=None)
c3d_req = CliRequirement('c3d', test_cmd='c3d', version_switch=None)


class FSLRequirement(CliRequirement):
    """
    Since FSL doesn't have a convenient '--version' switch we can use, we need
    to interrogate the FSLDIR to find the fslversion
    """

    def __init__(self):
        super().__init__('fsl', 'fslinfo', version_switch=None)

    def detect_version(self):
        """
        As FSL doesn't have a simple way of printing the version, the best
        we can do is to read the copyright statement of a shell command.

        Note that this doesn't pick up the micro release
        """
        try:
            fsl_dir = os.environ['FSLDIR']
        except KeyError:
            raise ArcanaRequirementNotFoundError(
                "Could not find FSL, 'FSLDIR' environment variable is not set")
        with open(op.join(fsl_dir, 'etc', 'fslversion'), 'r') as f:
            contents = f.read()
        return self.parse_version(contents.strip())


# Create an instance of the FSLRequirement class
fsl_req = FSLRequirement()


# Matlab package requirements

class SpmRequirement(MatlabPackageRequirement):

    def __init__(self):
        super().__init__('spm', test_func='spm_authors')

    def parse_help_text(self, help_text):
        match = re.search(
            r'Copyright \(C\) [\d\-\, ]*(?<!\d)(\d+) Wellcome Trust Centre',
            help_text)
        if match is None:
            raise ArcanaRequirementVersionNotDectableError(
                "Could not parse year of copyright from spm_authors in order "
                "to determine the version of {}".format(self))
        copyright_year = match.group(1)
        if copyright_year == '2010':
            version = 8
        elif copyright_year == '2012':
            version = 12
        else:
            raise ArcanaRequirementVersionNotDectableError(
                "Do not know the version of SPM corresponding to the year of "
                "copyright of {}".format(copyright_year))
        return version


spm_req = SpmRequirement()
sti_req = MatlabPackageRequirement('sti', test_func='V_SHARP')
# noddi_req = MatlabPackageRequirement('noddi')


# Python package requirements

sklearn_req = PythonPackageRequirement('sklearn')
pydicom_req = PythonPackageRequirement('pydicom')
scipy_req = PythonPackageRequirement('scipy')

# mrtrix0_3_req = CliRequirement('mrtrix', min_version=(0, 3, 12),
#                                max_version=(0, 3, 15), test_cmd='mrconvert')
# mrtrix_req.v('3.0') = CliRequirement('mrtrix', min_version=(3, 0, 0),
#                              test_cmd='mrconvert')
# fsl5_req = CliRequirement('fsl', min_version=(5, 0, 8), test_cmd='fsl')
# fsl_req.v('5.0.9') = CliRequirement('fsl', min_version=(5, 0, 9),
#                             max_version=(5, 0, 9), test_cmd='fsl')
# fsl_req.v('5.0.10') = CliRequirement('fsl', min_version=(5, 0, 10),
#                             max_version=(5, 0, 10), test_cmd='fsl')
# ants_req.v('2.0') = CliRequirement('ants', min_version=(2, 0),
#                            test_cmd='antsRegistration')
# ants_req.v('1.9') = CliRequirement('ants', min_version=(1, 9),
#                             test_cmd='antsRegistration')
# spm_req.v(12) = MatlabRequirement('spm', min_version=(12, 0),
#                               test_func=None)
# freesurfer_req.v('5.3') = CliRequirement('freesurfer', min_version=(5, 3))
# matlab_req.v('R2014a') = CliRequirement('matlab', min_version=(2014, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')
# matlab_req.v('R2015a') = CliRequirement('matlab', min_version=(2015, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')
# noddi_req = MatlabRequirement('noddi', min_version=(0, 9)),
# niftimatlab_req = MatlabRequirement('niftimatlib', (1, 2))
# dcm2niix_req.v('1.0.2') = CliRequirement('dcm2niix', min_version=(1, 0, 2),
#                               test_cmd='dcm2niix')
# fix_req.v('1.0') = CliRequirement('fix', min_version=(1, 0))
# afni_req.v('16.2.10') = CliRequirement('afni', min_version=(16, 2, 10))
# mricrogl_req = CliRequirement('mricrogl', min_version=(1, 0, 20170207))
# stir_req.v('3.0') = CliRequirement('stir', min_version=(3, 0))
# c3d_req.v('1.1.0') = CliRequirement('c3d', min_version=(1, 1, 0))

# matlab_req.v('R2015a') = Requirement('matlab', min_version=(2015, 'a'),
#                                 version_split=matlab_version_split,
#                                 test_cmd='matlab')
