from __future__ import absolute_import
import os
import os.path as op
import stat
import logging
from bids.layout import BIDSLayout
import grabbit
from arcana.data import Fileset, Field
from arcana.pipeline import Record
from arcana.exceptions import ArcanaUsageError
from arcana.repository import DirectoryRepository, Tree
import banana.file_format  # @UnusedImport
from banana.data.bids import BidsFileset


logger = logging.getLogger('arcana')


class BidsRepository(DirectoryRepository):
    """
    An 'Repository' class for directories on the local file system organised
    into sub-directories by subject and then visit.

    Parameters
    ----------
    root_dir : str
        The path to the root of the BidsRepository
    """

    type = 'bids'

    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._layout = BIDSLayout(root_dir)

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def layout(self):
        return self._layout

    def __repr__(self):
        return "BidsRepository(root_dir='{}')".format(self.root_dir)

    def __eq__(self, other):
        try:
            return self.root_dir == other.root_dir
        except AttributeError:
            return False

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
        fields = []
        records = []
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
                        name=item.entities['type'],
                        path=op.join(item.dirname, item.filename),
                        subject_id=subject_id, visit_id=visit_id,
                        repository=self,
                        modality=item.entities.get('modality', None),
                        run=item.entities.get('run', None),
                        task=item.entities.get('task', None))
                    filesets.append(fileset)
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
