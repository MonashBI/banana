from arcana.exceptions import ArcanaSelectorError, ArcanaUsageError
from arcana.data.selector import FilesetSelector
from arcana.data.item import Fileset
from arcana.data.file_format import FileFormat
from arcana.utils import split_extension


class BaseBids(object):

    def __init__(self, modality, run, task):
        self._modality = modality
        self._run = run
        self._task = task

    def __eq__(self, other):
        return (self.type == other.type and
                self.task == other.task and
                self.modality == other.modality and
                self.run == other.run)

    def __hash__(self):
        return (hash(self.type) ^
                hash(self.task) ^
                hash(self.modality) ^
                hash(self.run))

    def initkwargs(self):
        dct = {}
        dct['task'] = self.task
        dct['modality'] = self.modality
        dct['run'] = self.run
        return dct

    @property
    def type(self):
        return self.name

    @property
    def modality(self):
        return self._modality

    @property
    def run(self):
        return self._run

    @property
    def task(self):
        return self._task


class BidsFileset(Fileset, BaseBids):
    """
    A representation of a fileset within the repository.

    Parameters
    ----------
    name : str
        The name of the fileset
    path : str | None
        The path to the fileset on the local system
    subject_id : int | str | None
        The id of the subject which the fileset belongs to
    visit_id : int | str | None
        The id of the visit which the fileset belongs to
    repository : BaseRepository
        The repository which the fileset is stored
    modality : str
        The BIDS modality
    run : str
        The BIDS run
    task : str
        The BIDS task
    checksums : dict[str, str]
        A checksums of all files within the fileset in a dictionary sorted by
        relative file paths
    """

    def __init__(self, type, path, subject_id, visit_id, repository,  # @ReservedAssignment @IgnorePep8
                 modality=None, run=None, task=None, checksums=None):
        Fileset.__init__(
            self,
            name=type,
            format=FileFormat.by_ext(split_extension(path)[1]),
            frequency='per_session',
            path=path,
            subject_id=subject_id,
            visit_id=visit_id,
            repository=repository,
            checksums=checksums)
        BaseBids.__init__(modality, run, task)


class BidsSelector(FilesetSelector, BaseBids):
    """
    A match object for matching filesets from their 'bids_attr'
    attribute

    Parameters
    ----------
    type : str
        Type of the fileset
    task : str
        The task the fileset belongs to
    modality : str
        Modality of the filesets
    format : FileFormat
        The file format of the fileset to match
    run : int
        Run number of the fileset
    """

    def __init__(self, type, format, task=None, modality=None, run=None):  # @ReservedAssignment @IgnorePep8
        FilesetSelector.__init__(
            self, type, format, pattern=None, frequency='per_session',   # @ReservedAssignment @IgnorePep8
            id=None, dicom_tags=None, is_regex=False,
            from_study=None)
        BaseBids.__init__(modality, run, task)

    def _filtered_matches(self, node):
        matches = [f for f in node.filesets if BaseBids.__eq__(self, f)]
        if not matches:
            raise ArcanaSelectorError(
                "No BIDS filesets for {} match {} found:\n{}"
                .format(node, self, '\n'.join(str(f) for f in node.filesets)))
        return matches

    def __repr__(self):
        return "{}(type={}, task={}, modality={})".format(
            self.__class__.__name__, self.type, self.task, self.modality)

    def __eq__(self, other):
        return (FilesetSelector.__eq__(self, other) and
                BaseBids.__eq__(self, other))

    def __hash__(self):
        return (FilesetSelector.__hash__(self) ^
                BaseBids.__hash__(self))

    def initkwargs(self):
        dct = FilesetSelector.initkwargs(self)
        dct.update(BaseBids.initkwargs(self))
        return dct


class BidsAssociatedSelector(FilesetSelector):
    """
    A match object for matching BIDS filesets that are associated with
    another BIDS filesets (e.g. field-maps, bvecs, bvals)

    Parameters
    ----------
    name : str
        Name of the associated fileset
    primary_match : BidsMatch
        The primary fileset which the fileset to match is associated with
    associated : str
        The name of the association between the fileset to match and the
        primary fileset
    fieldmap_type : str
        Key of the return fieldmap dictionary (if association=='fieldmap'
    order : int
        Order of the fieldmap dictionary that you want to match
    """

    VALID_ASSOCIATIONS = ('fieldmap', 'bvec', 'bval')

    def __init__(self, name, primary_match, format, association,  # @ReservedAssignment @IgnorePep8
                 fieldmap_type=None, order=0):
        FilesetSelector.__init__(
            self, name, format, pattern=None, frequency='per_session',   # @ReservedAssignment @IgnorePep8
            id=None, order=order, dicom_tags=None, is_regex=False,
            from_study=None)
        self._primary_match = primary_match
        self._association = association
        if fieldmap_type is not None and association != 'fieldmap':
            raise ArcanaUsageError(
                "'fieldmap_type' (provided to '{}' match) "
                "is only valid for 'fieldmap' "
                "associations (not '{}')".format(name, association))
        self._fieldmap_type = fieldmap_type

    def __repr__(self):
        return ("{}(name={}, primary_match={}, format={}, association={}, "
                "fieldmap_type\{}, order={})".format(
                    self.name, self.primary_match, self.format,
                    self.association, self.fieldmap_type,
                    self.order))

    @property
    def primary_match(self):
        return self._primary_match

    @property
    def association(self):
        return self._association

    @property
    def fieldmap_type(self):
        return self._fieldmap_type

    def _bind_node(self, node):
        primary_match = self._primary_match._bind_node(node)
        layout = self.study.repository.layout
        if self._association == 'fieldmap':
            matches = layout.get_fieldmap(primary_match.path,
                                          return_list=True)
            try:
                match = matches[0]
            except IndexError:
                raise ArcanaSelectorError(
                    "Provided order to associated BIDS fileset match "
                    "{} is out of range")
        elif self._association == 'bvec':
            match = layout.get_bvec(primary_match.path)
        elif self._association == 'bval':
            match = layout.get_bval(primary_match.path)
        return match
        return matches

    def __eq__(self, other):
        return (FilesetSelector.__eq__(self, other) and
                self.primary_match == other.primary_match and
                self.association == other.association and
                self.fieldmap_type == other.fieldmap_type)

    def __hash__(self):
        return (FilesetSelector.__hash__(self) ^
                hash(self.primary_match) ^
                hash(self.association) ^
                hash(self.fieldmap_type))

    def initkwargs(self):
        dct = FilesetSelector.initkwargs(self)
        dct['primary_match'] = self.primary_match
        dct['association'] = self.association
        dct['fieldmap_type'] = self.fieldmap_type
        return dct
