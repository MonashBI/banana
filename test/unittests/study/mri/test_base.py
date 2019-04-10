#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from banana.study.mri.base import MriStudy  # @IgnorePep8
from banana.utils.testing import PipelineTester, TEST_CACHE_DIR  # @IgnorePep8 @Reimport
from arcana.repository.xnat import XnatRepo  # @IgnorePep8


class TestBase(PipelineTester):

    study_class = MriStudy
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANAMRI',
                        cache_dir=TEST_CACHE_DIR)
    parameters = {}

    def test_all(self):
        self.all_tests()
