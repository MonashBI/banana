import os
import os.path as op
import nianalysis
from nianalysis.requirement import fsl5_req
from arcana.data import Fileset, FilesetCollection


class BaseAtlas():

    def __call__(self, study):
        name, path = self._get_path(study)
        return FilesetCollection(name, [Fileset.from_path(path)],
                                 frequency='per_study')

    def _get_path(self, study):  # @UnusedVariable
        """
        Parameters
        ----------
        study : Study
            The study to return the atlas for

        Returns
        -------
        name : str
            A name for the atlas
        path : str
            The path to the atlas file
        """
        return NotImplementedError


class FslAtlas(BaseAtlas):
    """
    Class to retrieve the path to an atlas shipped with a FSL installation

    Parameters
    ----------
    name : str
        Name of atlas or family of atlases
    resolution : str | float
        The resolution of the atlas to use. Can either be a fixed floating
        point value or the name of a parameter in the study to draw the
        value from
    dataset : str | None
        Name of the dataset (i.e. 'brain', 'brain_mask', 'eye_mask', 'edges')
        will be append to the atlas name using '_' as a delimeter
    sub_path : str
        Relative path to atlas directory from FSL 'data' directory
    """

    def __init__(self, name, resolution=1, dataset=None,
                 sub_path=['standard']):
        self._name = name
        self._resolution = resolution
        self._dataset = dataset
        self._sub_path = sub_path

    def _get_path(self, study):
        # If resolution is a string then it is assumed to be a parameter name
        # of the study
        atlas_name = self._name
        if isinstance(self._resolution, str):
            res = study.parameter(self._resolution)
        else:
            res = self._resolution
        atlas_name += '_{}mm'.format(res)
        if self._dataset is not None:
            atlas_name += '_' + self._dataset
        fsl_ver = study.satisfier.load(fsl5_req)
        atlas_dir = op.join(os.environ['FSLDIR'], 'data', *self._sub_path)
        study.satisfier.unload(fsl_ver)
        return atlas_name, op.join(atlas_dir, atlas_name + '.nii.gz')


class QsmAtlas(BaseAtlas):
    """

    """

    BASE_PATH = op.abspath(op.join(op.dirname(nianalysis.__file__), 'atlases'))

    def __init__(self, name):
        self._name = name

    def _get_path(self, study):  # @UnusedVariable
        return self._name, op.join(self.BASE_PATH, self._name + '.nii.gz')
