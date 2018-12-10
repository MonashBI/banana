from copy import copy
from arcana.study import (
    Study as ArcanaStudy, MultiStudy, StudyMetaClass, MultiStudyMetaClass)  # @UnusedImport @IgnorePep8
from arcana.exceptions import ArcanaSelectorMissingMatchError


# Extend Arcana Study class to support implicit BIDS selectors

class Study(ArcanaStudy):

    def __init__(self, name, repository, processor, inputs=None,
                 bids_task=None, **kwargs):
        self._bids_task = bids_task
        # Attempt to preload default bids inputs
        bids_inputs = {}
        if repository.type == 'bids':
            try:
                # If the study has the attribute default bids inputs then
                # then check to see if they are present in the repository
                default_bids_inputs = self.default_bids_inputs
            except AttributeError:
                pass
            else:
                tree = repository.cached_tree()
                # Check to see which bids_inputs are present in the repo
                for bids_input in default_bids_inputs:
                    # If bids_task is passed as an input to the study and the
                    # task isn't explicitly set in the default bids input copy
                    # it across here.
                    if self.bids_task and bids_input.task:
                        bids_input = copy(bids_input)
                        bids_input._task = self.bids_task
                    try:
                        bids_input.match(tree)
                    except ArcanaSelectorMissingMatchError:
                        pass
                    else:
                        bids_inputs[bids_input.name] = bids_input
        # Update the inputs dictionary (which at this point just contains the
        # present defaults) with the explicit inputs.
        if inputs is not None:
            bids_inputs.update(inputs)
        ArcanaStudy.__init__(self, name, repository, processor, bids_inputs,
                             **kwargs)

    @property
    def bids_task(self):
        return self._bids_task
