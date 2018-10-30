from unittest import TestCase
from arcana.environment import Requirement, CliRequirement
from arcana.exception import (
    ArcanaRequirementNotFoundError, ArcanaRequirementVersionNotDectableError)
import banana.requirement


for req_name in dir(banana.requirement):
    req = getattr(banana.requirement, req_name)
    if not isinstance(req, CliRequirement) or req.name == 'matlab':
        continue
    try:
        version = req.detect_version()
    except ArcanaRequirementNotFoundError:
        print("Could not find requirement for {}".format(req))
    except ArcanaRequirementVersionNotDectableError:
        print("No version information is available for {}".format(req))
    else:
        print("Found {} version for {}".format(version, req))
