from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function

workflow = Workflow('test_workflow')
workflow.base_dir = "tmp"

subjects = Node(
    name='subjects',
    interface=IdentityInterface(
        fields=['subject_id']))

subjects.iterables = ('subject_id', ['subject1', 'subject2', 'subject3'])

visits = Node(
    name='visits',
    interface=IdentityInterface(
        fields=['visit_id']))

visits.iterables = ('visit_id', ['visit1', 'visit2'])


# merging subjects and visits ids
def merge(subject_id, visit_id):
    return (subject_id, visit_id)


sessions = Node(Function(input_names=["subject_id", "visit_id"],
                         output_names=["pair"],
                         function=merge, name='sessions'), name="session")


# for join node: create a list from all elements
def create_list(pair):
    return list(pair)


join_list = JoinNode(Function(input_names=['pair'],
                              output_names=['pairs_list'],
                              function=create_list),
                     name='list', joinsource='subjects', joinfield=['pair'])


# for join node: concatenate all lists from previous join node
def concatenate(pairs_list):
    out = []
    for el in pairs_list:
        out += el
    return out


join_concate = JoinNode(Function(input_names=['pairs_list'],
                                 output_names=['all_pairs'],
                                 function=concatenate),
                        name='con', joinsource='visits',
                        joinfield=['pairs_list'])

workflow.connect(subjects, 'subject_id', sessions, 'subject_id')
workflow.connect(visits, 'visit_id', sessions, 'visit_id')
workflow.connect(sessions, 'pair', join_list, 'pair')
workflow.connect(join_list, 'pairs_list', join_concate, "pairs_list")

result = workflow.run()
