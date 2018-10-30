# from unittest import TestCase
from arcana.environment import Requirement
from arcana.exception import (
    ArcanaRequirementNotFoundError, ArcanaRequirementVersionNotDectableError)
import banana.requirement


requirement_attrs = [getattr(banana.requirement, r)
                     for r in dir(banana.requirement)]

for req in requirement_attrs:
    if not isinstance(req, Requirement):
        continue
    try:
        version = req.detect_version()
    except ArcanaRequirementNotFoundError:
        print("Could not find match for {}".format(req))
    except ArcanaRequirementVersionNotDectableError:
        print("No version information is available for {}".format(req))
    else:
        print("Found {} version for {}".format(version, req))
