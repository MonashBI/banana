from arcana.study import (
    Study as ArcanaStudy, MultiStudy, StudyMetaClass, MultiStudyMetaClass)  # @UnusedImport @IgnorePep8
from banana.exceptions import BananaUsageError


# Extend Arcana Study class to support implicit BIDS selectors

class Study(ArcanaStudy):

    def __init__(self, name, repository, processor, inputs=None, **kwargs):
        if inputs is None:
            if repository.type == 'bids':
                try:
                    inputs = self.bids_inputs
                except AttributeError:
                    raise BananaUsageError(
                        "No 'bids_inputs' attribute in {} class, explicit "
                        "selections need to be provided when it is "
                        "instantiated ".format(self.__type__.__name__))
            else:
                raise BananaUsageError(
                    "Inputs selections, matching data in the repository with "
                    "specifications in the class need to be provided when "
                    "initialising a {} objects".format(self.__type__.__name__))
        ArcanaStudy.__init__(self, name, repository, processor, inputs,
                             **kwargs)
