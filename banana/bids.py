from arcana.exceptions import (
    ArcanaSelectorMissingMatchError, ArcanaUsageError)
from banana.exceptions import BananaUsageError
from arcana.data.selector import FilesetSelector
from arcana.data.item import Fileset
from arcana.data.file_format import FileFormat
from arcana.utils import split_extension
import os
import os.path as op
import stat
import logging
from bids.layout import BIDSLayout
from arcana.repository import DirectoryRepository
import banana.file_format  # @UnusedImport


logger = logging.getLogger('arcana')


class BidsRepository(DirectoryRepository):
    """
    A repository class for BIDS datasets

    Parameters
    ----------
    root_dir : str
        The path to the root of the BidsRepository
    """

    type = 'bids'

    def __init__(self, root_dir, **kwargs):
        DirectoryRepository.__init__(self, root_dir, depth=2, **kwargs)
        self._layout = BIDSLayout(root_dir)

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def layout(self):
        return self._layout

    def __repr__(self):
        return "BidsRepository(root_dir='{}')".format(self.root_dir)

    def __hash__(self):
        return super().__hash__()

    def find_data(self, subject_ids=None, visit_ids=None):
        """
        Return subject and session information for a project in the local
        repository

        Parameters
        ----------
        subject_ids : list(str)
            List of subject IDs with which to filter the tree with. If None all
            are returned
        visit_ids : list(str)
            List of visit IDs with which to filter the tree with. If None all
            are returned

        Returns
        -------
        project : arcana.repository.Tree
            A hierarchical tree of subject, session and fileset information for
            the repository
        """
        filesets = []
        all_subjects = self.layout.get_subjects()
        all_visits = self.layout.get_sessions()
        for item in self.layout.get(return_type='object'):
            if not hasattr(item, 'entities') or not item.entities.get('type',
                                                                      False):
                logger.warning("Skipping unrecognised file '{}' in BIDS tree"
                               .format(op.join(item.dirname, item.filename)))
                continue  # Ignore hidden file
            try:
                subject_ids = [item.entities['subject']]
            except KeyError:
                # If item exists in top-levels of in the directory structure
                # it is inferred to exist for all subjects in the tree
                subject_ids = all_subjects
            try:
                visit_ids = [item.entities['session']]
            except KeyError:
                # If item exists in top-levels of in the directory structure
                # it is inferred to exist for all visits in the tree
                visit_ids = all_visits
            for subject_id in subject_ids:
                for visit_id in visit_ids:
                    fileset = BidsFileset(
                        path=op.join(item.dirname, item.filename),
                        type=item.entities['type'],
                        subject_id=subject_id, visit_id=visit_id,
                        repository=self,
                        modality=item.entities.get('modality', None),
                        task=item.entities.get('task', None))
                    filesets.append(fileset)
        # Get derived filesets, fields and records using the same method using
        # the method in the DirectoryRepository base class
        derived_filesets, fields, records = super().find_data(
            subject_ids=subject_ids, visit_ids=visit_ids)
        filesets.extend(derived_filesets)
        return filesets, fields, records

    def fileset_path(self, item, fname=None):
        if not item.derived:
            raise ArcanaUsageError(
                "Can only get automatically get path to derived filesets not "
                "{}".format(item))
        if fname is None:
            fname = item.fname
        if item.subject_id is not None:
            subject_id = item.subject_id
        else:
            subject_id = self.SUMMARY_NAME
        if item.visit_id is not None:
            visit_id = item.visit_id
        else:
            visit_id = self.SUMMARY_NAME
        sess_dir = op.join(self.root_dir, 'derivatives', item.from_study,
                           'sub-{}'.format(subject_id),
                           'sess-{}'.format(visit_id))
        # Make session dir if required
        if not op.exists(sess_dir):
            os.makedirs(sess_dir, stat.S_IRWXU | stat.S_IRWXG)
        return op.join(sess_dir, fname)

    def _extract_ids_from_path(self, path_parts, *args, **kwargs):  # @UnusedVariable @IgnorePep8
        if len(path_parts) != 4 or path_parts[0] != 'derivatives':
            return None
        from_study, subj, sess = path_parts[1:]
        subj_id = subj[len('sub-'):]
        visit_id = sess[len('sess-'):]
        return subj_id, visit_id, from_study


class BaseBidsFileset(object):

    derived = False

    def __init__(self, type, modality, task):  # @ReservedAssignment
        self._modality = modality
        self._type = type
        self._task = task

    def __eq__(self, other):
        return (self.type == other.type and
                self.task == other.task and
                self.modality == other.modality)

    def __hash__(self):
        return (hash(self.type) ^
                hash(self.task) ^
                hash(self.modality))

    def initkwargs(self):
        dct = {}
        dct['type'] = self.type
        dct['task'] = self.task
        dct['modality'] = self.modality
        return dct

    @property
    def modality(self):
        return self._modality

    @property
    def task(self):
        return self._task

    @property
    def type(self):
        return self._type


class BidsFileset(Fileset, BaseBidsFileset):
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
    task : str
        The BIDS task
    checksums : dict[str, str]
        A checksums of all files within the fileset in a dictionary sorted by
        relative file paths
    """

    def __init__(self, path, type, subject_id, visit_id, repository,  # @ReservedAssignment @IgnorePep8
                 modality=None, task=None, checksums=None):
        Fileset.__init__(
            self,
            name=op.basename(path),
            format=FileFormat.by_ext(split_extension(path)[1]),
            frequency='per_session',
            path=path,
            subject_id=subject_id,
            visit_id=visit_id,
            repository=repository,
            checksums=checksums)
        BaseBidsFileset.__init__(self, type, modality, task)

    def __repr__(self):
        return ("{}(type={}, task={}, modality={}, format={}, subj={}, vis={})"
                .format(self.__class__.__name__, self.type, self.task,
                        self.modality, self.format.name, self.subject_id,
                        self.visit_id))


class BidsSelector(FilesetSelector, BaseBidsFileset):
    """
    A match object for matching filesets from their BIDS attributes and file
    format. If any of the provided attributes are None, then that attribute
    is omitted from the match

    Parameters
    ----------
    spec_name : str
        Name of the spec to match
    type : str
        Type of the fileset
    format : FileFormat
        The file format of the fileset to match
    task : str
        The task the fileset belongs to
    modality : str
        Modality of the filesets
    """

    def __init__(self, spec_name, type, format=None, task=None, modality=None,  # @ReservedAssignment @IgnorePep8
                 **kwargs):
        FilesetSelector.__init__(
            self, spec_name, pattern=None, format=format, frequency='per_session',   # @ReservedAssignment @IgnorePep8
            **kwargs)
        BaseBidsFileset.__init__(self, type, modality, task)

    def _filtered_matches(self, node):
        matches = [
            f for f in node.filesets
            if (isinstance(f, BidsFileset) and
                self.type == f.type and
                (self.modality is None or self.modality == f.modality) and
                (self.task is None or self.task == f.task) and
                (self.format is None or self.format == f.format))]
        if not matches:
            raise ArcanaSelectorMissingMatchError(
                "No BIDS filesets for {} match {} found:\n{}"
                .format(node, self, '\n'.join(str(f) for f in node.filesets)))
        return matches

    def __repr__(self):
        return ("{}(spec_name='{}', type={}, format={}, modality={}, task={})"
                .format(
                    self.__class__.__name__,
                    self.spec_name,
                    self.type,
                    self._format.name if self._format is not None else None,
                    self.modality, self.task))

    def __eq__(self, other):
        return (FilesetSelector.__eq__(self, other) and
                BaseBidsFileset.__eq__(self, other))

    def __hash__(self):
        return (FilesetSelector.__hash__(self) ^
                BaseBidsFileset.__hash__(self))

    def initkwargs(self):
        dct = FilesetSelector.initkwargs(self)
        dct.update(BaseBidsFileset.initkwargs(self))
        return dct

    def _check_args(self):
        pass  # Disable check for either pattern or ID in base class

    @BaseBidsFileset.task.setter
    def task(self, task):
        self._task = task


class BidsAssociatedSelector(FilesetSelector):
    """
    A match object for matching BIDS filesets that are associated with
    another BIDS filesets (e.g. field-maps, bvecs, bvals)

    Parameters
    ----------
    name : str
        Name of the associated fileset
    primary : BidsSelector
        A selector to select the primary fileset which the associated fileset
        is associated with
    association : str
        The name of the association between the fileset to match and the
        primary fileset, can be one of 'bvec', 'bval', 'phase1', 'phase2',
        'phasediff', 'epi' or 'fieldmap'
    fieldmap_order : int
        If there are more than one field-maps associated with the primary
        fileset, which one to return
    """

    VALID_ASSOCIATIONS = ('bvec', 'bval', 'phase1', 'phase2', 'phasediff',
                          'epi', 'fieldmap')

    def __init__(self, spec_name, primary, association, format=None,   # @ReservedAssignment @IgnorePep8
                 fieldmap_order=0, **kwargs):
        FilesetSelector.__init__(self, spec_name, format,
                                 frequency='per_session', **kwargs)
        self._primary = primary
        if association not in self.VALID_ASSOCIATIONS:
            raise BananaUsageError(
                "Invalid association '{}' passed to BidsAssociatedSelector, "
                "can be one of '{}'".format(
                    association, "', '".join(self.VALID_ASSOCIATIONS)))
        self._association = association
        self._fieldmap_order = fieldmap_order

    def __eq__(self, other):
        return (FilesetSelector.__eq__(self, other) and
                self.primary == other.primary and
                self.format == other.format and
                self.association == other.association and
                self.fieldmap_type == other.fieldmap_type)

    def __hash__(self):
        return (FilesetSelector.__hash__(self) ^
                hash(self.primary) ^
                hash(self.format) ^
                hash(self.association) ^
                hash(self.fieldmap_order))

    def initkwargs(self):
        dct = FilesetSelector.initkwargs(self)
        dct['primary'] = self.primary
        dct['format'] = self.primary
        dct['association'] = self.association
        dct['fieldmap_order'] = self.fieldmap_order
        return dct

    def __repr__(self):
        return ("{}(spec_name={}, primary={}, format={}, association={}, "
                "fieldmap_order={})".format(
                    type(self).__name__,
                    self.spec_name, self.primary,
                    self._format.name if self._format is not None else None,
                    self.association, self.fieldmap_order))

    def bind(self, study, spec_name=None, **kwargs):
        # We need to access a bound primary selector when matching the
        # associated selector so we set the bound version temporarily to
        # self._primary before winding it back after we have done the bind
        unbound_primary = self._primary
        self._primary = self._primary.bind(study, **kwargs)
        bound = super().bind(study, spec_name=spec_name, **kwargs)
        self._primary = unbound_primary
        return bound

    @property
    def primary(self):
        return self._primary

    @property
    def association(self):
        return self._association

    @property
    def fieldmap_order(self):
        return self._fieldmap_order

    @property
    def task(self):
        return self.primary.task

    @task.setter
    def task(self, task):
        self.primary.task = task

    def match_node(self, node):
        primary_match = self.primary.match_node(node)
        layout = self.primary.repository.layout
        if self.association == 'bvec':
            path = layout.get_bvec(primary_match.path)
        elif self.association == 'bval':
            path = layout.get_bval(primary_match.path)
        else:
            fieldmaps = layout.get_fieldmap(primary_match.path,
                                            return_list=True)
            if not len(fieldmaps):
                raise ArcanaSelectorMissingMatchError(
                    "No matches for field-map '{}' associated with {}"
                    .format(self.association, primary_match))
            try:
                fieldmap = fieldmaps[self.fieldmap_order]
            except IndexError:
                raise ArcanaSelectorMissingMatchError(
                    "Index of field-map ({}) is outside range of available "
                    "field-maps ({}) for {}"
                    .format(self.fieldmap_order,
                            ', '.join(str(f) for f in fieldmaps),
                            primary_match))
            path = fieldmap[self.association]
        return Fileset.from_path(path, format=self._format,
                                 repository=self.primary.repository,
                                 subject_id=node.subject_id,
                                 visit_id=node.visit_id)
