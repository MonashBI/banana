#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from banana.study.mri.base import MriStudy  # @IgnorePep8
from banana.utils.testing import PipelineTester  # @IgnorePep8 @Reimport
from arcana.repository.xnat import XnatRepo  # @IgnorePep8


class TestBase(PipelineTester):

    study_class = MriStudy
    ref_repo = XnatRepo()
