#!/usr/bin/env python
import os
import os.path as op
import shutil
from nipype import config
config.enable_debug_mode()
from arcana import LocalRepository, LinearRunner, DatasetMatch  # @IgnorePep8
from nianalysis.file_format import dicom_format, nifti_gz_format  # @IgnorePep8
from nianalysis.study.ct import CtStudy  # @IgnorePep8
from unittest import TestCase  # @IgnorePep8 @Reimport

home_dir = os.environ['HOME']
work_dir = op.join(home_dir, 'work')
repo_dir = op.join(home_dir, 'Data', 'andrii')

shutil.rmtree(work_dir, ignore_errors=True)
os.makedirs(work_dir)


class TestCt(TestCase):

    def test_registration(self):
        study = CtStudy(
            'ct_study',
            runner=LinearRunner(work_dir),
            repository=LocalRepository(repo_dir),
            inputs=[DatasetMatch('ct_umap', nifti_gz_format, 'ct'),
                    DatasetMatch('dicom_ref', dicom_format,
                                 'dicom-reference')],
            enforce_inputs=False)
        ct_reg = list(study.data('ct_reg_dicom'))[0]
        print("Converted file can be found at '{}'".format(ct_reg.path))
