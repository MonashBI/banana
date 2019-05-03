from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface


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
        fields=['subject_id', 'visit_id']))

join_subjects = JoinNode(
    name='join_subjects',
    interface=IdentityInterface(
        fields=['subject_ids', 'visit_ids']),
    joinsource='subjects',
    joinfield=['subject_ids', 'visit_ids'])

join_visits = JoinNode(
    name='join_visits',
    interface=IdentityInterface(
        fields=['subject_ids', 'visit_ids']),
    joinsource='visits',
    joinfield=['subject_ids', 'visit_ids'])

workflow.connect(subjects, 'subject_id', sessions, 'subject_id')
workflow.connect(visits, 'visit_id', sessions, 'visit_id')
workflow.connect(sessions, 'subject_id', join_subjects, 'subject_ids')
workflow.connect(sessions, 'visit_id', join_subjects, 'visit_ids')
workflow.connect(join_subjects, 'subject_id', join_visits, 'subject_ids')
workflow.connect(join_subjects, 'visit_id', join_visits, 'visit_ids')

result = workflow.run()
