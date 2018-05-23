from arcana.requirement import Requirement, matlab_version_split


mrtrix0_3_req = Requirement('mrtrix', min_version=(0, 3, 12),
                            max_version=(0, 3, 15))
mrtrix3_req = Requirement('mrtrix', min_version=(3, 0, 0))
fsl5_req = Requirement('fsl', min_version=(5, 0, 8))
fsl509_req = Requirement('fsl', min_version=(5, 0, 9),
                         max_version=(5, 0, 9))
fsl510_req = Requirement('fsl', min_version=(5, 0, 10),
                         max_version=(5, 0, 10))
ants2_req = Requirement('ants', min_version=(2, 0))
ants19_req = Requirement('ants', min_version=(1, 9))
spm12_req = Requirement('spm', min_version=(12, 0))
freesurfer_req = Requirement('freesurfer', min_version=(5, 3))
matlab2014_req = Requirement('matlab', min_version=(2014, 'a'),
                             version_split=matlab_version_split)
matlab2015_req = Requirement('matlab', min_version=(2015, 'a'),
                             version_split=matlab_version_split)
noddi_req = Requirement('noddi', min_version=(0, 9)),
niftimatlab_req = Requirement('niftimatlib', (1, 2))
dcm2niix_req = Requirement('dcm2niix', min_version=(1, 0, 2))
fix_req = Requirement('fix', min_version=(1, 0))
afni_req = Requirement('afni', min_version=(16, 2, 10))
mricrogl_req = Requirement('mricrogl', min_version=(1, 0, 20170207))
stir_req = Requirement('stir', min_version=(3, 0))
