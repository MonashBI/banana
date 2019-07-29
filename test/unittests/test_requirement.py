# from unittest import TestCase
from arcana.environment import BaseRequirement, ModulesEnv
from arcana.exceptions import (
    ArcanaRequirementNotFoundError, ArcanaVersionNotDetectableError)
import banana.requirement

from banana.requirement import mrtrix_req

env = ModulesEnv()

env.satisfy(mrtrix_req.v('3.0rc3'))

# requirement_attrs = [getattr(banana.requirement, r)
#                      for r in dir(banana.requirement)]

# for req in requirement_attrs:
#     if not isinstance(req, BaseRequirement):
#         continue
#     try:
#         version = req.detect_version()
#     except ArcanaRequirementNotFoundError:
#         print("Could not find a version of {}".format(req))
#     except ArcanaVersionNotDetectableError:
#         print("No version information is available for {}".format(req))
#     else:
#         print("Found {} version for {}".format(version, req))
