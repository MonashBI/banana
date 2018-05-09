#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.data_format import nifti_gz_format  # @IgnorePep8
from mbianalysis.study.mri.base import MRIStudy  # @IgnorePep8
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from arcana.study import (  # @IgnorePep8
    MultiStudy, MultiStudyMetaClass, SubStudySpec)


class TestCoregStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('ref', MRIStudy),
        SubStudySpec('tocoreg', MRIStudy,
                     {'ref_brain': 'coreg_ref'})]


class TestMRI(TestCase):

    def test_coreg_and_brain_mask(self):
        study = self.create_study(
            TestCoregStudy, 'coreg_and_mask_study', inputs=[
                DatasetMatch('ref_primary', nifti_gz_format, 'mprage'),
                DatasetMatch('tocoreg_primary', nifti_gz_format,
                             'flair')])
        coreg_brain = study.data('tocoreg_coreg_brain')[0]
        self.assertDatasetsEqual(coreg_brain,
                                 self.reference('coreg_brain'))
