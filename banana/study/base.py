from arcana.study import (
    Study as ArcanaStudy, MultiStudy, StudyMetaClass, MultiStudyMetaClass)  # @UnusedImport @IgnorePep8


# Extend Arcana Study class to support implicit BIDS selectors

class Study(ArcanaStudy):

    def __init__(self, name, repository, processor, inputs=None,
                 bids_task=None, **kwargs):
        # IDs need to be set here before the study tree is accessed
        self._bids_task = bids_task
        # Attempt to preload default bids inputs
        bids_inputs = {}
        if repository.type == 'bids':
            try:
                # If the study has the attribute default bids inputs then
                # then check to see if they are present in the repository
                bids_inputs = self.default_bids_inputs
            except AttributeError:
                pass
        # Update the inputs dictionary (which at this point just contains the
        # present defaults) with the explicit inputs.
        if inputs is not None:
            bids_inputs.update(inputs)
        ArcanaStudy.__init__(self, name, repository, processor, bids_inputs,
                             **kwargs)

    @property
    def bids_task(self):
        return self._bids_task
