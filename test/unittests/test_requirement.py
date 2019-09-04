from unittest import TestCase
from arcana.environment import BaseRequirement
from arcana.exceptions import (
    ArcanaRequirementNotFoundError, ArcanaVersionNotDetectableError)
import banana.requirement


requirement_attrs = [getattr(banana.requirement, r)
                     for r in dir(banana.requirement)]


class TestRequirement(TestCase):

    def test_requirements(self):
        for req in requirement_attrs:
            if not isinstance(req, BaseRequirement):
                continue
            try:
                version = req.detect_version()
            except ArcanaRequirementNotFoundError:
                print("Could not find a version of {}".format(req))
            except ArcanaVersionNotDetectableError:
                print("No version information is available for {}".format(req))
            else:
                print("Found {} version for {}".format(version, req))
