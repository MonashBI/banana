#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetInput  # @IgnorePep8
from banana.file_format import nifti_gz_format  # @IgnorePep8
from banana.study.mri.base import MriStudy  # @IgnorePep8
from banana.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from arcana.study import (  # @IgnorePep8
    MultiStudy, MultiStudyMetaClass, SubStudySpec)


class TestCoregStudy(MultiStudy, metaclass=MultiStudyMetaClass):

    add_sub_study_specs = [
        SubStudySpec('ref', MriStudy),
        SubStudySpec('tocoreg', MriStudy,
                     {'ref_brain': 'coreg_ref'})]


class TestMRI(TestCase):

    def test_coreg_and_brain_mask(self):
        study = self.create_study(
            TestCoregStudy, 'coreg_and_mask_study', inputs=[
                FilesetInput('ref_primary', 'mprage', nifti_gz_format),
                FilesetInput('tocoreg_primary', nifti_gz_format,
                             'flair')])
        coreg_brain = list(study.data('tocoreg_coreg_brain'))[0]
        self.assertFilesetsEqual(coreg_brain,
                                 self.reference('coreg_brain'))
