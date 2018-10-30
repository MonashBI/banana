from unittest import TestCase
from arcana.environment import Requirement, CliRequirement
import banana.requirement


for req_name in dir(banana.requirement):
    req = getattr(banana.requirement, req_name)
    if not isinstance(req, CliRequirement) or req.name == 'matlab':
        continue
    version = req.detect_version()
    print("Found {} version for {}".format(version, req))
