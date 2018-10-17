import os
import os.path as op
from arcana.exception import ArcanaError
import banana
from banana.requirement import fsl5_req


def nth(i):
    "Returns 1st, 2nd, 3rd, 4th, etc for a given number"
    if i == 1:
        s = '1st'
    elif i == 2:
        s = '2nd'
    elif i == 3:
        s = '3rd'
    else:
        s = '{}th'.format(i)
    return s


def get_fsl_reference_path():
    return op.join(os.environ['FSLDIR'], 'data', 'standard')


def get_atlas_path(name, fileset='brain', resolution='1mm'):
    """
    Returns the path to the atlas (or atlas mask) in the arcana repository

    Parameters
    ----------
    name : str
        Name of the Atlas, can be one of ('mni_nl6')
    atlas_type : str
        Whether to return the brain mask or the full atlas, can be one of
        'image', 'mask'
    """
    if name == 'MNI152':
        # MNI ICBM 152 non-linear 6th Generation Symmetric Average Brain
        # Stereotaxic Registration Model (http://nist.mni.mcgill.ca/?p=858)
        if resolution not in ['0.5mm', '1mm', '2mm']:
            raise ArcanaError(
                "Invalid resolution for MNI152, '{}', can be one of '0.5mm', "
                "'1mm' or '2mm'".format(resolution))
        if fileset == 'image':
            path = op.join(get_fsl_reference_path(),
                                'MNI152_T1_{}.nii.gz'.format(resolution))
        elif fileset == 'mask':
            path = op.join(get_fsl_reference_path(),
                                'MNI152_T1_{}_brain_mask.nii.gz'
                                .format(resolution))
        elif fileset == 'mask_dilated':
            if resolution != '2mm':
                raise ArcanaError(
                    "Dilated MNI masks are not available for {} resolution "
                    .format(resolution))
            path = op.join(get_fsl_reference_path(),
                                'MNI152_T1_{}_brain_mask_dil.nii.gz'
                                .format(resolution))
        elif fileset == 'brain':
            path = op.join(get_fsl_reference_path(),
                                'MNI152_T1_{}_brain.nii.gz'
                                .format(resolution))
        else:
            raise ArcanaError("Unrecognised fileset '{}'"
                                  .format(fileset))
    else:
        raise ArcanaError("Unrecognised atlas name '{}'"
                              .format(name))
    return op.abspath(path)
