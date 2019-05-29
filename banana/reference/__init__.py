import os
import os.path as op
from copy import copy
import banana
from banana.requirement import fsl_req
from arcana.exceptions import ArcanaError
from arcana import Fileset, FilesetCollection


class BaseReference():

    frequency = 'per_study'

    def __init__(self, format, name=None):  # @ReservedAssignment
        self._name = name
        self._format = format
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

    def __hash__(self):
        return hash(self.name)

    @property
    def collection(self):
        return FilesetCollection(
            self.name,
            [Fileset.from_path(self.path, frequency=self.frequency)],
            format=self._format,
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


class FslReferenceData(BaseReference):
    """
    Class to retrieve the path to an atlas shipped with a FSL installation

    Parameters
    ----------
    atlas_name : str
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

    def __init__(self, atlas_name, format, name=None, resolution=1,  # @ReservedAssignment @IgnorePep8
                 dataset=None, sub_path=['standard']):
        super().__init__(format, name)
        self._atlas_name = atlas_name
        self._resolution = resolution
        self._dataset = dataset
        self._sub_path = tuple(sub_path.split('/') if isinstance(sub_path, str)
                               else sub_path)

    @property
    def path(self):
        # If resolution is a string then it is assumed to be a parameter name
        # of the study
        if isinstance(self._resolution, str):
            resolution = getattr(self.study, self._resolution)
        else:
            resolution = self._resolution
        full_atlas_name = '{}_{}mm'.format(self._atlas_name, resolution)
        if self._dataset is not None:
            full_atlas_name += '_' + self._dataset
        fsl_ver = self.study.environment.satisfy(fsl_req.v('5.0.8'))[0]
        if hasattr(self.study.environment, 'load'):
            self.study.environment.load(fsl_ver)
            fsl_dir = os.environ['FSLDIR']
            self.study.environment.unload(fsl_ver)
        else:
            fsl_dir = os.environ['FSLDIR']  # Static environments
        return op.join(fsl_dir, 'data', *self._sub_path,
                       full_atlas_name + '.nii.gz')

    def translate(self, substudy_spec):
        """
        Translate resolution parameter name if used to namespace of multi-study

        Parameters
        ----------
        substudy_spec : SubStudySpec
            The sub-study that the spec belongs to
        """
        if isinstance(self._resolution, str):
            self._resolution = substudy_spec.map(self._resolution)

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self._atlas_name == other._atlas_name and
            self._resolution == other._resolution and
            self._dataset == other._dataset and
            self._sub_path == other._sub_path)

    def __hash__(self):
        return (super().__hash__() ^
                hash(self._atlas_name) ^
                hash(self._resolution) ^
                hash(self._dataset) ^
                hash(self._sub_path))

    @property
    def _error_msg_loc(self):
        return "'{}' FSL atlas passed to '{}' in {} ".format(
            self._atlas_name, self.name, self.study)


class LocalReferenceData(BaseReference):
    """
    Several atlases used in the composite-vein analysis in the T2* study,
    stored within the banana package.

    Parameters
    ----------
    atlas_name : str
        Base name of the atlas file (i.e. without extension) in the 'atlases'
        directory
    """

    BASE_PATH = op.abspath(op.join(op.dirname(__file__), 'data'))

    def __init__(self, atlas_name, format, name=None):  # @ReservedAssignment
        super().__init__(format, name)
        self._atlas_name = atlas_name

    @property
    def path(self):  # @UnusedVariable
        return op.join(self.BASE_PATH, self._atlas_name + '.nii.gz')

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._atlas_name == other._atlas_name)

    def __hash__(self):
        return (super().__hash__() ^
                hash(self._atlas_name))
