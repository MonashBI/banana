from unittest import TestCase
from arcana.environment import BaseRequirement
from arcana.exceptions import (
    ArcanaRequirementNotFoundError, ArcanaVersionNotDetectableError)
from banana.utils.testing import TEST_ENV
import banana.requirement


all_requirements = {
    req.name: req for req in (getattr(banana.requirement, attr)
                              for attr in dir(banana.requirement))
    if isinstance(req, BaseRequirement)}

dependencies = {
    'fix': ['fsl'],
    'sti': ['matlab']}


class TestRequirement(TestCase):

    def test_requirements(self):
        """
        Test that all requirements can be satisfied (from their base versions)
        """
        for requirement in all_requirements.values():
            to_load = []

            def push_deps(req):
                "Recursively add requirement to load list incl. deps"
                for dep in dependencies.get(req.name, []):
                    push_deps(all_requirements[dep])
                to_load.append(req.base_version)

            push_deps(requirement)

            TEST_ENV.satisfy(*to_load)
