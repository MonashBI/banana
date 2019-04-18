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
            bids_inputs = {}
            for inpt in self.default_bids_inputs:
                if inpt.task is None and bids_task is not None:
                    inpt = copy(inpt)
                    inpt.task = self.bids_task
                bids_inputs[inpt.spec_name] = inpt
            # Combine explicit inputs with defaults, overriding any with
            # matching spec names
            bids_inputs.update(inputs)
            inputs = bids_inputs
        # Update the inputs di
        ArcanaStudy.__init__(self, name, repository, processor, inputs,
                             **kwargs)

    @property
    def bids_task(self):
        return self._bids_task
