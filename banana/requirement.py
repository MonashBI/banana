from arcana.environment.requirement import (
    CLIRequirement, MatlabRequirement, matlab_version_split)

mrtrix0_3_req = CLIRequirement('mrtrix', min_version=(0, 3, 12),
                               max_version=(0, 3, 15), test_cmd='mrconvert')
mrtrix3_req = CLIRequirement('mrtrix', min_version=(3, 0, 0),
                             test_cmd='mrconvert')
fsl5_req = CLIRequirement('fsl', min_version=(5, 0, 8), test_cmd='fsl')
fsl509_req = CLIRequirement('fsl', min_version=(5, 0, 9),
                            max_version=(5, 0, 9), test_cmd='fsl')
fsl510_req = CLIRequirement('fsl', min_version=(5, 0, 10),
                            max_version=(5, 0, 10), test_cmd='fsl')
ants2_req = CLIRequirement('ants', min_version=(2, 0),
                           test_cmd='antsRegistration')
ants19_req = CLIRequirement('ants', min_version=(1, 9),
                            test_cmd='antsRegistration')
spm12_req = MatlabRequirement('spm', min_version=(12, 0),
                              test_func=None)
freesurfer_req = CLIRequirement('freesurfer', min_version=(5, 3))
matlab2014_req = CLIRequirement('matlab', min_version=(2014, 'a'),
                                version_split=matlab_version_split,
                                test_cmd='matlab')
matlab2015_req = CLIRequirement('matlab', min_version=(2015, 'a'),
                                version_split=matlab_version_split,
                                test_cmd='matlab')
noddi_req = MatlabRequirement('noddi', min_version=(0, 9)),
niftimatlab_req = MatlabRequirement('niftimatlib', (1, 2))
dcm2niix_req = CLIRequirement('dcm2niix', min_version=(1, 0, 2),
                              test_cmd='dcm2niix')
fix_req = CLIRequirement('fix', min_version=(1, 0))
afni_req = CLIRequirement('afni', min_version=(16, 2, 10))
mricrogl_req = CLIRequirement('mricrogl', min_version=(1, 0, 20170207))
stir_req = CLIRequirement('stir', min_version=(3, 0))
c3d_req = CLIRequirement('c3d', min_version=(1, 1, 0))
