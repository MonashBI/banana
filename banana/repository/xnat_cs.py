import os
import os.path as op
import logging
import json
import xnat
from arcana.data import Fileset, Field
from arcana.pipeline.provenance import Record
from arcana.exceptions import (
    ArcanaRepositoryError)
from arcana.utils import get_class_info, split_extension
from arcana.repository import LocalFileSystemRepo
from banana.exceptions import BananaUsageError


logger = logging.getLogger('arcana')

COMMAND_INPUT_TYPES = {
    bool: 'bool',
    str: 'string',
    int: 'number',
    float: 'number'}

class XnatCSRepo(LocalFileSystemRepo):
    """
    A 'Repository' class for data stored within a XNAT repository and accessed
    via the XNAT container service.

    Parameters
    ----------
    root_dir : str (path)
        Path to local directory containing data
    """

    type = 'xnat_cs'
    SUMMARY_NAME = '__ALL__'
    FIELDS_FNAME = 'fields.json'
    PROV_DIR = '__prov__'
    LOCK_SUFFIX = '.lock'
    MAX_DEPTH = 2

    def __init__(self, project_id, session_id=None):
        super().__init__()
        self.project_id = project_id
        self.session_id = session_id
        self.xnat_host = os.environ['XNAT_HOST']
        self.token = os.environ['XNAT_USER']
        self.secret = os.environ['XNAT_PASS']

    def __eq__(self, other):
        try:
            return self.project_uri == other.project_uri
        except AttributeError:
            return False


    @property
    def prov(self):
        prov = {
            'type': get_class_info(type(self)),
            'xnat_host': self.xnat_host,
            'project_id': self.project_id}
        return prov

    def __hash__(self):
        return hash(tuple(self.prov.items()))

    # root_dir=None, all_from_analysis=None,
    def find_data(self, dataset, subject_ids=None, visit_ids=None, **kwargs):
        """
        Find all data within a repository, registering filesets, fields and
        provenance with the found_fileset, found_field and found_provenance
        methods, respectively

        Parameters
        ----------
        dataset : Dataset
            A dataset from which the root directory will be extracted from its
            name
        subject_ids : list(str)
            List of subject IDs with which to filter the tree with. If
            None all are returned
        visit_ids : list(str)
            List of visit IDs with which to filter the tree with. If
            None all are returned

        Returns
        -------
        filesets : list[Fileset]
            All the filesets found in the repository
        fields : list[Field]
            All the fields found in the repository
        records : list[Record]
            The provenance records found in the repository
        """
        all_filesets = []
        all_fields = []
        all_records = []

        with xnat.connect(self.xnat_host, user=self.token,
                          password=self.secret) as xlogin:
            xproject = xlogin.projects[self.project_id]
            if self.session_id:
                xsessions = [xproject.experiments[self.session_id]]
            else:
                xsessions = xproject.experiment.values()

        for xsession in xsessions:
            xsubject = xsession.subject()
        # if root_dir is None:
        input_dir = dataset.name
        for session_path, dirs, files in os.walk(root_dir):
            relpath = op.relpath(session_path, root_dir)
            path_parts = relpath.split(op.sep) if relpath != '.' else []
            ids = self._extract_ids_from_path(dataset.depth, path_parts, dirs,
                                              files)
            if ids is None:
                continue
            subj_id, visit_id, from_analysis = ids
            # if all_from_analysis is not None:
            #     if from_analysis is not None:
            #         raise ArcanaRepositoryError(
            #             "Found from_analysis sub-directory '{}' when global "
            #             "from analysis '{}' was passed".format(
            #                 from_analysis, all_from_analysis))
            #     else:
            #         from_analysis = all_from_analysis
            # Check for summaries and filtered IDs
            if subj_id == self.SUMMARY_NAME:
                subj_id = None
            elif subject_ids is not None and subj_id not in subject_ids:
                continue
            if visit_id == self.SUMMARY_NAME:
                visit_id = None
            elif visit_ids is not None and visit_id not in visit_ids:
                continue
            # Map IDs into ID space of analysis
            subj_id = dataset.map_subject_id(subj_id)
            visit_id = dataset.map_visit_id(visit_id)
            # Determine frequency of session|summary
            if (subj_id, visit_id) == (None, None):
                frequency = 'per_dataset'
            elif subj_id is None:
                frequency = 'per_visit'
            elif visit_id is None:
                frequency = 'per_subject'
            else:
                frequency = 'per_session'
            filtered_files = self._filter_files(files, session_path)
            for fname in filtered_files:
                basename = split_extension(fname)[0]
                all_filesets.append(
                    Fileset.from_path(
                        op.join(session_path, fname),
                        frequency=frequency,
                        subject_id=subj_id, visit_id=visit_id,
                        dataset=dataset,
                        from_analysis=from_analysis,
                        potential_aux_files=[
                            f for f in filtered_files
                            if (split_extension(f)[0] == basename
                                and f != fname)],
                        **kwargs))
            for fname in self._filter_dirs(dirs, session_path):
                all_filesets.append(
                    Fileset.from_path(
                        op.join(session_path, fname),
                        frequency=frequency,
                        subject_id=subj_id, visit_id=visit_id,
                        dataset=dataset,
                        from_analysis=from_analysis,
                        **kwargs))
            if self.FIELDS_FNAME in files:
                with open(op.join(session_path,
                                  self.FIELDS_FNAME), 'r') as f:
                    dct = json.load(f)
                all_fields.extend(
                    Field(name=k, value=v, frequency=frequency,
                          subject_id=subj_id, visit_id=visit_id,
                          dataset=dataset, from_analysis=from_analysis,
                          **kwargs)
                    for k, v in list(dct.items()))
            if self.PROV_DIR in dirs:
                if from_analysis is None:
                    raise ArcanaRepositoryError(
                        "Found provenance directory in session directory (i.e."
                        " not in analysis-specific sub-directory)")
                base_prov_dir = op.join(session_path, self.PROV_DIR)
                for fname in os.listdir(base_prov_dir):
                    all_records.append(Record.load(
                        split_extension(fname)[0],
                        frequency, subj_id, visit_id, from_analysis,
                        op.join(base_prov_dir, fname)))
        return all_filesets, all_fields, all_records

    def fileset_path(self, item, dataset=None, fname=None):
        pass

    def fields_json_path(self, field, dataset=None):
        return self.fileset_path(field, fname=self.FIELDS_FNAME,
                                 dataset=dataset)

    def prov_json_path(self, record, dataset):
        return self.fileset_path(record,
                                 dataset=dataset,
                                 fname=op.join(self.PROV_DIR,
                                               record.pipeline_name + '.json'))

    @classmethod
    def command_json(cls, image_name, analysis_cls, inputs, derivatives,
                     parameters, desc, frequency='per_session',
                     docker_index="https://index.docker.io/v1/"):

        if frequency != 'per_session':
            raise NotImplementedError(
                "Support for frequencies other than '{}' haven't been "
                "implemented yet".format(frequency))
        try:
            analysis_name, version = image_name.split('/')[1].split(':')
        except (IndexError, ValueError):
            raise BananaUsageError(
                "The Docker organisation and tag needs to be provided as part "
                "of the image, e.g. australianimagingservice/dwiqa:0.1")

        cmd_inputs = []
        input_names = []
        for inpt in inputs:
            input_name = inpt if isinstance(inpt, str) else inpt[0]
            input_names.append(input_name)
            spec = analysis_cls.data_spec(input_name)
            desc = spec.desc if spec.desc else ""
            if spec.is_fileset:
                desc = ("Scan match: {} [SCAN_TYPE [ORDER [TAG=VALUE, ...]]]"
                        .format(desc))
            else:
                desc = "Field match: {} [FIELD_NAME]".format(desc)
            cmd_inputs.append({
                "name": input_name,
                "description": desc,
                "type": "string",
                "default-value": "",
                "required": True,
                "user-settable": True,
                "replacement-key": "#{}_INPUT#".format(input_name.upper())})

        for param in parameters:
            spec = analysis_cls.param_spec(param)
            desc = "Parameter: " + spec.desc
            if spec.choices:
                desc += " (choices: {})".format(','.join(spec.choices))

            cmd_inputs.append({
                "name": param,
                "description": desc,
                "type": COMMAND_INPUT_TYPES[spec.dtype],
                "default-value": (spec.default
                                  if spec.default is not None else ""),
                "required": spec.default is None,
                "user-settable": True,
                "replacement-key": "#{}_PARAM#".format(param.upper())})

        cmd_inputs.append(
            {
                "name": "project-id",
                "description": "Project ID",
                "type": "string",
                "required": True,
                "user-settable": False,
                "replacement-key": "#PROJECT_ID#"
            })


        cmdline = (
            "banana derive /input {cls} {name} {derivs} {inputs} {params}"
            " --scratch /work --repository xnat_cs #PROJECT_URI#"
            .format(
                cls='.'.join((analysis_cls.__module__, analysis_cls.__name__)),
                name=analysis_name,
                derivs=' '.join(derivatives),
                inputs=' '.join('-i {} #{}_INPUT#'.format(i, i.upper())
                                for i in input_names),
                params=' '.join('-p {} #{}_PARAM#'.format(p, p.upper())
                                for p in parameters)))

        if frequency == 'per_session':
            cmd_inputs.append(
                {
                    "name": "session-id",
                    "description": "",
                    "type": "string",
                    "required": True,
                    "user-settable": False,
                    "replacement-key": "#SESSION_ID#"
                })
            cmdline += "#SESSION_ID# --session_ids #SESSION_ID# "

        return {
            "name": analysis_name,
            "description": desc,
            "label": analysis_name,
            "version": version,
            "schema-version": "1.0",
            "image": image_name,
            "index": docker_index,
            "type": "docker",
            "command-line": cmdline,
            "override-entrypoint": True,
            "mounts": [
                {
                    "name": "in",
                    "writable": False,
                    "path": "/input"
                },
                {
                    "name": "output",
                    "writable": True,
                    "path": "/output"
                },
                {
                    "name": "work",
                    "writable": True,
                    "path": "/work"
                }
            ],
            "ports": {},
            "inputs": cmd_inputs,
            "outputs": [
                {
                    "name": "output",
                    "description": "Derivatives",
                    "required": True,
                    "mount": "out",
                    "path": None,
                    "glob": None
                },
                {
                    "name": "working",
                    "description": "Working directory",
                    "required": True,
                    "mount": "work",
                    "path": None,
                    "glob": None
                }
            ],
            "xnat": [
                {
                    "name": analysis_name,
                    "description": desc,
                    "contexts": ["xnat:imageSessionData"],
                    "external-inputs": [
                        {
                            "name": "session",
                            "description": "Imaging session",
                            "type": "Session",
                            "matcher": None,
                            "default-value": None,
                            "required": True,
                            "replacement-key": None,
                            "sensitive": None,
                            "provides-value-for-command-input": None,
                            "provides-files-for-command-mount": "in",
                            "via-setup-command": None,
                            "user-settable": None,
                            "load-children": True
                        }
                    ],
                    "derived-inputs": [
                        {
                            "name": "session-id",
                            "type": "string",
                            "required": True,
                            "load-children": True,
                            "derived-from-wrapper-input": "session",
                            "derived-from-xnat-object-property": "id",
                            "provides-value-for-command-input": "session-id"
                        },
                        {
                            "name": "subject",
                            "type": "Subject",
                            "required": True,
                            "user-settable": False,
                            "load-children": True,
                            "derived-from-wrapper-input": "session"
                        },
                        {
                            "name": "project-id",
                            "type": "string",
                            "required": True,
                            "load-children": True,
                            "derived-from-wrapper-input": "subject",
                            "derived-from-xnat-object-property": "id",
                            "provides-value-for-command-input": "subject-id"
                        }
                    ],
                    "output-handlers": [
                        {
                            "name": "output-resource",
                            "accepts-command-output": "output",
                            "via-wrapup-command": None,
                            "as-a-child-of": "session",
                            "type": "Resource",
                            "label": "Derivatives",
                            "format": None
                        },
                        {
                            "name": "working-resource",
                            "accepts-command-output": "working",
                            "via-wrapup-command": None,
                            "as-a-child-of": "session",
                            "type": "Resource",
                            "label": "Work",
                            "format": None
                        }
                    ]
                }
            ]
        }



# ref_command = {
#   "name": "bids-mriqc",
#   "description": "Runs the MRIQC BIDS App",
#   "version": "1.1",
#   "schema-version": "1.0",
#   "image": "poldracklab/mriqc:0.15.2",
#   "type": "docker",
#   "command-line": "mriqc /input /output participant group --no-sub -w /work -v #FLAGS#",
#   "override-entrypoint": True,
#   "mounts": [
#     {
#       "name": "in",
#       "writable": false,
#       "path": "/input"
#     },
#     {
#       "name": "out",
#       "writable": true,
#       "path": "/output"
#     },
#     {
#       "name": "work",
#       "writable": true,
#       "path": "/work"
#     }
#   ],
#   "environment-variables": {},
#   "ports": {},
#   "inputs": [
#     {
#       "name": "flags",
#       "label": "MRIQC flags",
#       "description": "Flags to pass to MRIQC",
#       "type": "string",
#       "matcher": null,
#       "default-value": "",
#       "required": false,
#       "replacement-key": "#FLAGS#",
#       "sensitive": false,
#       "command-line-flag": null,
#       "command-line-separator": null,
#       "true-value": null,
#       "false-value": null,
#       "select-values": [],
#       "multiple-delimiter": null
#     }
#   ],
#   "outputs": [
#     {
#       "name": "output",
#       "description": "Output QC files",
#       "required": true,
#       "mount": "out",
#       "path": null,
#       "glob": null
#     },
#     {
#       "name": "working",
#       "description": "Working QC files",
#       "required": true,
#       "mount": "work",
#       "path": null,
#       "glob": null
#     }
#   ],
#   "xnat": [
#     {
#       "name": "bids-mriqc-project",
#       "label": null,
#       "description": "Run the MRIQC BIDS App with a project mounted",
#       "contexts": [
#         "xnat:projectData"
#       ],
#       "external-inputs": [
#         {
#           "name": "project",
#           "label": null,
#           "description": "Project",
#           "type": "Project",
#           "matcher": null,
#           "default-value": null,
#           "required": true,
#           "replacement-key": null,
#           "sensitive": null,
#           "provides-value-for-command-input": null,
#           "provides-files-for-command-mount": "in",
#           "via-setup-command": "radiologicskate/xnat2bids-setup:1.3:xnat2bids",
#           "user-settable": null,
#           "load-children": false
#         }
#       ],
#       "derived-inputs": [],
#       "output-handlers": [
#         {
#           "name": "output-resource",
#           "accepts-command-output": "output",
#           "via-wrapup-command": null,
#           "as-a-child-of": "project",
#           "type": "Resource",
#           "label": "MRIQC",
#           "format": null
#         },
#         {
#           "name": "working-resource",
#           "accepts-command-output": "working",
#           "via-wrapup-command": null,
#           "as-a-child-of": "project",
#           "type": "Resource",
#           "label": "MRIQC-wrk",
#           "format": null
#         }
#       ]
#     },
#     {
#       "name": "bids-mriqc-session",
#       "label": null,
#       "description": "Run the MRIQC BIDS App with a session mounted",
#       "contexts": [
#         "xnat:imageSessionData"
#       ],
#       "external-inputs": [
#         {
#           "name": "session",
#           "label": null,
#           "description": "Input session",
#           "type": "Session",
#           "matcher": null,
#           "default-value": null,
#           "required": true,
#           "replacement-key": null,
#           "sensitive": null,
#           "provides-value-for-command-input": null,
#           "provides-files-for-command-mount": "in",
#           "via-setup-command": "radiologicskate/xnat2bids-setup:1.3:xnat2bids",
#           "user-settable": null,
#           "load-children": false
#         }
#       ],
#       "derived-inputs": [],
#       "output-handlers": [
#         {
#           "name": "output-resource",
#           "accepts-command-output": "output",
#           "via-wrapup-command": null,
#           "as-a-child-of": "session",
#           "type": "Resource",
#           "label": "MRIQC",
#           "format": null
#         },
#         {
#           "name": "working-resource",
#           "accepts-command-output": "working",
#           "via-wrapup-command": null,
#           "as-a-child-of": "session",
#           "type": "Resource",
#           "label": "MRIQC-wrk",
#           "format": null
#         }
#       ]
#     }
#   ]
# }


# ids_command_json = {
#     "name": "Sample",
#     "description": "Sample",
#     "label": "sample-id-extration",
#     "version": "1.0",
#     "schema-version": "1.0",
#     "image": "busybox:latest",
#     "type": "docker",
#     "command-line": "echo '#PROJECT_ID# #SUBJECT_ID# #SESSION_ID#'",
#     "override-entrypoint": true,
#     "mounts": [
#         {
#             "name": "output",
#             "writable": true,
#             "path": "/output"
#         }
#     ],
#     "environment-variables": {
#         "PROJECT_ID": "#PROJECT_ID#",
#         "SUBJECT_ID": "#SUBJECT_ID#",
#         "SESSION_ID": "#SESSION_ID#"
#     },
#     "ports": {},
#     "inputs": [
#         {
#             "name": "session-id",
#             "description": "",
#             "type": "string",
#             "required": true,
#             "replacement-key": "#SESSION_ID#"
#         },
#         {
#             "name": "subject-id",
#             "description": "",
#             "type": "string",
#             "required": true,
#             "replacement-key": "#SUBJECT_ID#"
#         },
#         {
#             "name": "project-id",
#             "description": "",
#             "type": "string",
#             "required": true,
#             "replacement-key": "#PROJECT_ID#"
#         }
#     ],
#     "outputs": [
#         {
#             "name": "output",
#             "description": "The sample output",
#             "required": true,
#             "mount": "output"
#         }
#     ],
#     "xnat": [
#         {
#             "name": "Sample_ID_extration",
#             "description": "Sample ID extration",
#             "label": "sample-id-extration",
#             "contexts": ["xnat:imageScanData"],
#             "external-inputs": [
#                 {
#                     "name": "scan",
#                     "description": "Image Scan",
#                     "type": "Scan",
#                     "matcher": null,
#                     "default-value": null,
#                     "required": true,
#                     "replacement-key": null,
#                     "sensitive": null,
#                     "provides-value-for-command-input": null,
#                     "provides-files-for-command-mount": null,
#                     "via-setup-command": null,
#                     "user-settable": null,
#                     "load-children": true
#                 }
#             ],
#             "derived-inputs": [
#                 {
#                     "name": "session",
#                     "type": "Session",
#                     "required": true,
#                     "user-settable": false,
#                     "load-children": false,
#                     "derived-from-wrapper-input": "scan"
#                 },
#                 {
#                     "name": "session-id",
#                     "type": "string",
#                     "required": true,
#                     "load-children": true,
#                     "derived-from-wrapper-input": "session",
#                     "derived-from-xnat-object-property": "id",
#                     "provides-value-for-command-input": "session-id"
#                 },
#                 {
#                     "name": "subject",
#                     "type": "Subject",
#                     "required": true,
#                     "user-settable": false,
#                     "load-children": true,
#                     "derived-from-wrapper-input": "session"
#                 },
#                 {
#                     "name": "subject-id",
#                     "type": "string",
#                     "required": true,
#                     "load-children": true,
#                     "derived-from-wrapper-input": "subject",
#                     "derived-from-xnat-object-property": "id",
#                     "provides-value-for-command-input": "subject-id"
#                 },
#                 {
#                     "name": "project",
#                     "type": "Project",
#                     "required": true,
#                     "user-settable": false,
#                     "load-children": true,
#                     "derived-from-wrapper-input": "session"
#                 },
#                 {
#                     "name": "project-id",
#                     "type": "string",
#                     "required": true,
#                     "load-children": true,
#                     "derived-from-wrapper-input": "project",
#                     "derived-from-xnat-object-property": "id",
#                     "provides-value-for-command-input": "project-id"
#                 }
#             ],
#             "output-handlers": [
#                 {
#                     "name": "output-resource",
#                     "accepts-command-output": "output",
#                     "via-wrapup-command": null,
#                     "as-a-child-of": "scan",
#                     "type": "Resource",
#                     "label": "OUTPUT",
#                     "format": null,
#                     "description": "An output resource."
#                 }
#             ]
#         }
#     ]
# }