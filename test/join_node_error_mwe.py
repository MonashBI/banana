from itertools import chain
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio
from nipype.interfaces.io import IOBase
from nipype.interfaces.base import (
    InputMultiPath, BaseInterfaceInputSpec, traits, TraitedSpec)


class Chain(IdentityInterface):

    def _list_outputs(self):
        outputs = super(Chain, self)._list_outputs()
        chained_outputs = {}
        for k, v in outputs.items():
            chained_outputs[k] = list(chain(*v))
        return chained_outputs


class SelectSessionInputSpec(BaseInterfaceInputSpec):
    inlist = InputMultiPath(
        traits.Any, mandatory=True, desc='List of items to select from')
    subject_ids = traits.List(traits.Str, mandatory=True,
                              desc=('List of subject IDs corresponding to the '
                                    'provided items'))
    visit_ids = traits.List(traits.Str, mandatory=True,
                            desc=('List of visit IDs corresponding to the '
                                  'provided items'))
    subject_id = traits.Str(mandatory=True, desc='Subject ID')
    visit_id = traits.Str(mandatory=True, desc='Visit ID')


class SelectSessionOutputSpec(TraitedSpec):
    out = traits.Any(desc='selected value')


class SelectSession(IOBase):
    """Basic interface class to select session from a list"""

    input_spec = SelectSessionInputSpec
    output_spec = SelectSessionOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        assert len(self.inputs.subject_ids) == len(self.inputs.inlist)
        assert len(self.inputs.visit_ids) == len(self.inputs.inlist)
        session_ids = list(zip(self.inputs.subject_ids, self.inputs.visit_ids))
        index = session_ids.index((self.inputs.subject_id,
                                   self.inputs.visit_id))
        outputs['out'] = self.inputs.inlist[index]
        return outputs


workflow = Workflow('test_workflow')

subjects = Node(
    name='subjects',
    interface=IdentityInterface(
        fields=['subject_id']))

subjects.iterables = ('subject_id', ['subject1', 'subject2', 'visit2'])

visits = Node(
    name='visits',
    interface=IdentityInterface(
        fields=['visit_id']))

visits.iterables = ('visit_id', ['visit1', 'visit2'])

sessions = Node(
    name='sessions',
    interface=IdentityInterface(
        fields=['image', 'subject_id', 'visit_id']))

datasource = Node(
    nio.DataGrabber(
        base_directory='/my-dataset',
        template='subject%s/visit_%s/image.nii.gz',
        infields=['subject_id', 'visit_id'],
        sort_filelist=True),
    name='datasource')

normaliser = Node(
    Normaliser(),
    name='normaliser')


join_subjects = JoinNode(
    name='join_subjects',
    interface=IdentityInterface(
        fields=['subject_ids', 'visit_ids']),
    joinsource='subjects',
    joinfield=['subject_ids', 'visit_ids'])

join_visits = JoinNode(
    name='join_visits',
    interface=Chain(
        fields=['subject_ids', 'visit_ids']),
    joinsource='visits',
    joinfield=['subject_ids', 'visit_ids'])

selector = Node(
    SelectSession(),
    name='selector')

...

Session specific workflow to follow

...

workflow.connect(subjects, 'subject_id', sessions, 'subject_id')
workflow.connect(visits, 'visit_id', sessions, 'visit_id')
workflow.connect(sessions, 'subject_id', datasource, 'subject_id')
workflow.connect(sessions, 'visit_id', datasource, 'visit_id')
workflow.connect(datasource, 'outfiles', normaliser, 'infiles')
workflow.connect(sessions, 'subject_id', join_subjects, 'subject_ids')
workflow.connect(sessions, 'visit_id', join_subjects, 'visit_ids')
workflow.connect(join_subjects, 'subject_id', join_visits, 'subject_ids')
workflow.connect(join_subjects, 'visit_id', join_visits, 'visit_ids')
workflow.connect(normaliser, 'outfiles', selector, 'inlist')
workflow.connect(join_visits, 'subject_ids', selector, 'subject_ids')
workflow.connect(join_visits, 'visit_ids', selector, 'visit_ids')
workflow.connect(sessions, 'subject_id', selector, 'subject_id')
workflow.connect(sessions, 'visit_id', selector, 'visit_id')


result = workflow.run()
