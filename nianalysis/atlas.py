import os
import os.path as op
import nianalysis
from nianalysis.requirement import fsl5_req
from arcana.exception import ArcanaError
from arcana.data import Fileset, FilesetCollection


class BaseAtlas():

    def __init__(self, name):
        self._name = name

    def bind(self, study):
        self._study = study
        return self

    @property
    def study(self):
        try:
            return self._study
        except AttributeError:
            raise ArcanaError(
                "Can't access study property as {} has not been bound"
                .format(self))

    @property
    def name(self):
        return self._name

    @property
    def collection(self):
        return FilesetCollection(self.name, [Fileset.from_path(self.path)],
                                 frequency='per_study')

    @property
    def format(self):
        return self.collection.format

    @property
    def path(self):  # @UnusedVariable
        return NotImplementedError

    def __repr__(self):
        return '{}(name={})'.format(type(self).__name__, self._name)

    def __eq__(self, other):
        return self._name == other._name


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
        super().__init__(name)
        self._resolution = resolution
        self._dataset = dataset
        self._sub_path = sub_path

    @property
    def path(self):
        # If resolution is a string then it is assumed to be a parameter name
        # of the study
        fsl_ver = self.study.satisfier.load(fsl5_req)
        atlas_dir = op.join(os.environ['FSLDIR'], 'data', *self._sub_path)
        self.study.satisfier.unload(fsl_ver)
        return op.join(atlas_dir, self.name + '.nii.gz')

    @property
    def name(self):
        "Append resolution and dataset to atlas name"
        atlas_name = self._name
        if isinstance(self._resolution, str):
            res = self.study.parameter(self._resolution)
        else:
            res = self._resolution
        atlas_name += '_{}mm'.format(res)
        if self._dataset is not None:
            atlas_name += '_' + self._dataset
        return atlas_name

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self._resolution == other._resolution and
            self._dataset == other._dataset and
            self._sub_path == other._sub_path)


class QsmAtlas(BaseAtlas):
    """

    """

    BASE_PATH = op.abspath(op.join(op.dirname(nianalysis.__file__), 'atlases'))

    @property
    def path(self):  # @UnusedVariable
        return op.join(self.BASE_PATH, self.name + '.nii.gz')
