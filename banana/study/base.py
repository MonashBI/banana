from copy import copy
from arcana.study import (
    Study as ArcanaStudy, MultiStudy, StudyMetaClass, MultiStudyMetaClass)  # @UnusedImport @IgnorePep8


# Extend Arcana Study class to support implicit BIDS selectors

# TODO: need to extend for MultiStudy's too

class Study(ArcanaStudy):

    def __init__(self, name, repository, processor, inputs=None,
                 bids_task=None, **kwargs):
        if inputs is None:
            inputs = {}
        elif not isinstance(inputs, dict):
            inputs = {i.spec_name: i for i in inputs}
        # IDs need to be set here before the study tree is accessed
        self._bids_task = bids_task
        # Attempt to preload default bids inputs
        if repository.type == 'bids' and hasattr(self, 'default_bids_inputs'):
            # If the study has the attribute default bids inputs then
            # then check to see if they are present in the repository
            bids_inputs = self.get_bids_inputs(bids_task)
            # Combine explicit inputs with defaults, overriding any with
            # matching spec names
            bids_inputs.update(inputs)
            inputs = bids_inputs
        # Update the inputs di
        ArcanaStudy.__init__(self, name, repository, processor, inputs,
                             **kwargs)

    @classmethod
    def get_bids_inputs(cls, task=None, repository=None):
        inputs = {}
        for inpt in cls.default_bids_inputs:
            inpt = copy(inpt)
            if inpt.task is None and task is not None:
                inpt.task = task
            inpt._repository = repository
            inputs[inpt.spec_name] = inpt
        return inputs

    @property
    def bids_task(self):
        return self._bids_task
