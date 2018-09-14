#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana import LocalRepository, LinearRunner, DatasetMatch  # @IgnorePep8
from nianalysis.file_format import dicom_format, nifti_gz_format  # @IgnorePep8
from nianalysis.study.ct import CtStudy  # @IgnorePep8
from unittest import TestCase  # @IgnorePep8 @Reimport


class TestCt(TestCase):

    def test_registration(self):
        study = CtStudy(
            'ct_study',
            runner=LinearRunner('/Users/apoz0003/work'),
            repository=LocalRepository('/Users/apoz0003/git/repo'),
            inputs=[DatasetMatch('ct_umap', nifti_gz_format, 'ct'),
                    DatasetMatch('dicom_ref', dicom_format, 'dicom-reference')],
            enforce_inputs=False)
        ct_reg = study.data('ct_reg_dicom')[0]
        print(ct_reg.path)
