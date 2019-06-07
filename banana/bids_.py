import os
import json
import os.path as op
import stat
import logging
from bids.layout import BIDSLayout
from arcana.exceptions import (
    ArcanaInputMissingMatchError, ArcanaUsageError)
from banana.exceptions import BananaUsageError
from arcana.data.input import InputFilesets
from arcana.data.item import Fileset
from arcana.utils import split_extension
from arcana.repository import BasicRepo
from banana.file_format import (
    nifti_gz_format, nifti_gz_x_format, fsl_bvecs_format, fsl_bvals_format,
    tsv_format, json_format)


logger = logging.getLogger('arcana')

BIDS_FORMATS = (nifti_gz_x_format, nifti_gz_format, fsl_bvecs_format,
                fsl_bvals_format, tsv_format, json_format)


def detect_format(path, aux_files):
    ext = split_extension(path)[1]
    aux_names = set(aux_files.keys())
    for frmt in BIDS_FORMATS:
        if frmt.extension == ext and set(frmt.aux_files.keys()) == aux_names:
            return frmt
    raise BananaUsageError(
        "No matching BIDS format matches provided path ({}) and aux files ({})"
        .format(path, aux_files))


class BidsRepo(BasicRepo):
    """
    A repository class for BIDS datasets

    Parameters
    ----------
    root_dir : str
        The path to the root of the BidsRepo
    """

    type = 'bids'

    def __init__(self, root_dir, **kwargs):
        BasicRepo.__init__(self, root_dir, depth=2, **kwargs)

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def derivatives_dir(self):
        return op.join(self.root_dir, 'derivatives')

    @property
    def metadata_dir(self):
        """
        A temporary dir where we write out combined JSON side cars to include
        in extended nifti filesets
        """
        return op.join(self.derivatives_dir, '__metadata__')

    @property
    def layout(self):
        return self._layout

    def __repr__(self):
        return "BidsRepo(root_dir='{}')".format(self.root_dir)

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
        layout = BIDSLayout(self.root_dir)
        all_subjects = layout.get_subjects()
        all_visits = layout.get_sessions()
        for item in layout.get(return_type='object'):
            if item.path.startswith(self.derivatives_dir):
                # We handle derivatives using the BasicRepo base
                # class methods
                continue
            if not hasattr(item, 'entities') or not item.entities.get('suffix',
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
                    aux_files = {}
                    metadata = layout.get_metadata(item.path)
                    if metadata and not item.path.endswith('.json'):
                        # Write out the combined JSON side cars to a temporary
                        # file to include in extended NIfTI filesets
                        metadata_path = op.join(
                            self.metadata_dir,
                            'sub-{}'.format(subject_id),
                            'ses-{}'.format(visit_id),
                            item.filename + '.json')
                        os.makedirs(op.dirname(metadata_path), exist_ok=True)
                        if not op.exists(metadata_path):
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f)
                        aux_files['json'] = metadata_path
                    fileset = BidsFileset(
                        path=op.join(item.dirname, item.filename),
                        type=item.entities['suffix'],
                        subject_id=subject_id, visit_id=visit_id,
                        repository=self,
                        modality=item.entities.get('modality', None),
                        task=item.entities.get('task', None),
                        aux_files=aux_files)
                    filesets.append(fileset)
        # Get derived filesets, fields and records using the same method using
        # the method in the BasicRepo base class
        derived_filesets, fields, records = super().find_data(
            subject_ids=subject_ids, visit_ids=visit_ids)
        filesets.extend(derived_filesets)
        return filesets, fields, records

    def fileset_path(self, fileset, fname=None):
        if not fileset.derived:
            raise ArcanaUsageError(
                "Can only get automatically get path to derived filesets not "
                "{}".format(fileset))
        if fname is None:
            fname = fileset.fname
        if fileset.subject_id is not None:
            subject_id = fileset.subject_id
        else:
            subject_id = self.SUMMARY_NAME
        if fileset.visit_id is not None:
            visit_id = fileset.visit_id
        else:
            visit_id = self.SUMMARY_NAME
        sess_dir = op.join(self.root_dir, 'derivatives', fileset.from_study,
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
    repository : Repository
        The repository which the fileset is stored
    modality : str
        The BIDS modality
    task : str
        The BIDS task
    checksums : dict[str, str]
        A checksums of all files within the fileset in a dictionary sorted by
        relative file paths
    aux_files : dict[str, str]
        A dictionary containing a mapping from a side car name to path of the
        file
    """

    def __init__(self, path, type, subject_id, visit_id, repository,  # @ReservedAssignment @IgnorePep8
                 modality=None, task=None, checksums=None, aux_files=None):
        Fileset.__init__(
            self,
            name=op.basename(path),
            frequency='per_session',
            format=detect_format(path, aux_files),
            path=path,
            subject_id=subject_id,
            visit_id=visit_id,
            repository=repository,
            checksums=checksums,
            aux_files=aux_files)
        BaseBidsFileset.__init__(self, type, modality, task)

    def __repr__(self):
        return ("{}(type={}, task={}, modality={}, format={}, subj={}, vis={})"
                .format(self.__class__.__name__, self.type, self.task,
                        self.modality, self.format.name, self.subject_id,
                        self.visit_id))


class BidsInput(InputFilesets, BaseBidsFileset):
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
    valid_formats : FileFormat | list(FileFormat)
        The file format of the fileset to match, or a list of valid formats
    task : str
        The task the fileset belongs to
    modality : str
        Modality of the filesets
    """

    def __init__(self, spec_name, type, format=None, task=None, modality=None,  # @ReservedAssignment @IgnorePep8
                 **kwargs):
        InputFilesets.__init__(
            self, spec_name, pattern=None, format=format,
            frequency='per_session', **kwargs)  # @ReservedAssignment @IgnorePep8
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
            raise ArcanaInputMissingMatchError(
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
        return (InputFilesets.__eq__(self, other) and
                BaseBidsFileset.__eq__(self, other))

    def __hash__(self):
        return (InputFilesets.__hash__(self) ^
                BaseBidsFileset.__hash__(self))

    def initkwargs(self):
        dct = InputFilesets.initkwargs(self)
        dct.update(BaseBidsFileset.initkwargs(self))
        return dct

    def _check_args(self):
        pass  # Disable check for either pattern or ID in base class

    @BaseBidsFileset.task.setter
    def task(self, task):
        self._task = task


class BidsAssocInput(InputFilesets):
    """
    A match object for matching BIDS filesets that are associated with
    another BIDS filesets (e.g. field-maps, bvecs, bvals)

    Parameters
    ----------
    name : str
        Name of the associated fileset
    primary : BidsInput
        A selector to select the primary fileset which the associated fileset
        is associated with
    association : str
        The name of the association between the fileset to match and the
        primary fileset, can be one of 'bvec', 'bval', 'phase1', 'phase2',
        'phasediff', 'epi' or 'fieldmap'
    type : int
        If there are more than one field-maps associated with the primary
        fileset, which one to return
    """

    VALID_ASSOCIATIONS = ('grads', 'phase', 'phasediff', 'epi', 'fieldmap')

    def __init__(self, spec_name, primary, association, type=None, format=None,   # @ReservedAssignment @IgnorePep8
                 **kwargs):
        InputFilesets.__init__(self, spec_name, format,
                                 frequency='per_session', **kwargs)
        self._primary = primary
        if association not in self.VALID_ASSOCIATIONS:
            raise BananaUsageError(
                "Invalid association '{}' passed to BidsAssocInput, "
                "can be one of '{}'".format(
                    association, "', '".join(self.VALID_ASSOCIATIONS)))
        self._association = association
        self._type = type

    def __eq__(self, other):
        return (InputFilesets.__eq__(self, other) and
                self.primary == other.primary and
                self.format == other.format and
                self.association == other.association and
                self._type == other._type)

    def __hash__(self):
        return (InputFilesets.__hash__(self) ^
                hash(self.primary) ^
                hash(self.format) ^
                hash(self.association) ^
                hash(self._type))

    def initkwargs(self):
        dct = InputFilesets.initkwargs(self)
        dct['primary'] = self.primary
        dct['format'] = self.primary
        dct['association'] = self.association
        dct['type'] = self._type
        return dct

    def __repr__(self):
        return ("{}(spec_name={}, primary={}, format={}, association={}, "
                "type={})".format(
                    type(self).__name__,
                    self.spec_name, self.primary,
                    self._format.name if self._format is not None else None,
                    self.association, self.type))

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
    def type(self):
        return self._type if self._type is not None else self.association

    @property
    def task(self):
        return self.primary.task

    @task.setter
    def task(self, task):
        self.primary.task = task

    def match_node(self, node):
        primary_match = self.primary.match_node(node)
        layout = self.primary.repository.layout
        if self.association == 'grads':
            if self.type == 'bvec':
                path = layout.get_bvec(primary_match.path)
            elif self.type == 'bval':
                path = layout.get_bval(primary_match.path)
            else:
                raise ArcanaUsageError(
                    "'{}' is not a valid type for '{}' associations"
                    .format(self.type, self.association))
        else:
            fieldmaps = layout.get_fieldmap(primary_match.path,
                                            return_list=True)
            try:
                fieldmap = next(f for f in fieldmaps
                                     if f['type'] == self.association)
            except StopIteration:
                raise ArcanaInputMissingMatchError(
                    "No \"{}\" field-maps associated with {} (found {})"
                    .format(self.association, primary_match,
                            ', '.join(f['type'] for f in fieldmaps)))
            try:
                path = fieldmap[self.type]
            except KeyError:
                raise ArcanaUsageError(
                    "'{}' is not a valid type for '{}' associations"
                    .format(self.type, self.association))
        return Fileset.from_path(path, format=self._format,
                                 repository=self.primary.repository,
                                 subject_id=node.subject_id,
                                 visit_id=node.visit_id)
