import os
import os.path as op
from copy import copy
import banana
from banana.requirement import fsl5_req
from arcana.exception import (
    ArcanaError, ArcanaNameError, ArcanaUsageError, ArcanaDesignError)
from arcana import Fileset, FilesetCollection, MultiStudy


class BaseAtlas():

    frequency = 'per_study'

    def __init__(self, name=None):
        self._name = name
        self._study = None

    def bind(self, study):
        bound = copy(self)
        bound._study = study
        return bound

    @property
    def study(self):
        if self._study is None:
            raise ArcanaError(
                "Can't access study property as {} has not been bound"
                .format(self))
        return self._study

    @property
    def name(self):
        if self._name is None:
            raise ArcanaError(
                "Name for atlas hasn't been set, it should be set in when "
                "it is passed as a default")
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def collection(self):
        return FilesetCollection(
            self.name,
            [Fileset.from_path(self.path, frequency=self.frequency)],
            frequency=self.frequency)

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

    def __init__(self, atlas_name, name=None, resolution=1, dataset=None,
                 sub_path=['standard']):
        super().__init__(name)
        self._atlas_name = atlas_name
        self._resolution = resolution
        self._dataset = dataset
        self._sub_path = sub_path

    @property
    def path(self):
        # If resolution is a string then it is assumed to be a parameter name
        # of the study
        if isinstance(self._resolution, str):
            resolution = self.study.parameter(self._resolution)
        else:
            resolution = self._resolution
        full_atlas_name = '{}_{}mm'.format(self._atlas_name, resolution)
        if self._dataset is not None:
            full_atlas_name += '_' + self._dataset
        fsl_ver = self.study.environment.load(fsl5_req)
        atlas_dir = op.join(os.environ['FSLDIR'], 'data', *self._sub_path)
        self.study.environment.unload(fsl_ver)
        return op.join(atlas_dir, full_atlas_name + '.nii.gz')

    def translate(self, sub_study_spec):
        """
        Translate resolution parameter name if used to namespace of multi-study

        Parameters
        ----------
        sub_study_spec : SubStudySpec
            The sub-study that the spec belongs to
        """
        if isinstance(self._resolution, str):
            self._resolution = sub_study_spec.inverse_map(self._resolution)

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self._atlas_name == other._atlas_name and
            self._resolution == other._resolution and
            self._dataset == other._dataset and
            self._sub_path == other._sub_path)

    @property
    def _error_msg_loc(self):
        return "'{}' FSL atlas passed to '{}' in {} ".format(
            self._atlas_name, self.name, self.study)


class LocalAtlas(BaseAtlas):
    """
    Several atlases used in the composite-vein analysis in the T2* study,
    stored within the banana package.

    Parameters
    ----------
    atlas_name : str
        Base name of the atlas file (i.e. without extension) in the 'atlases'
        directory
    """

    BASE_PATH = op.abspath(op.join(op.dirname(banana.__file__), 'atlases'))

    def __init__(self, atlas_name, name=None):
        super().__init__(name)
        self._atlas_name = atlas_name

    @property
    def path(self):  # @UnusedVariable
        return op.join(self.BASE_PATH, self._atlas_name + '.nii.gz')

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._atlas_name == other._atlas_name)
