from arcana.testing import BaseTestCase, BaseMultiSubjectTestCase
from arcana.study.base import Study, StudyMetaClass
from arcana.dataset import DatasetSpec, DatasetMatch
from nianalysis.data_format import (
    dicom_format)


class TestMatchStudy(Study):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('gre_phase', dicom_format),
        DatasetSpec('gre_mag', dicom_format)]

    def dummy_pipeline1(self):
        pass

    def dummy_pipeline2(self):
        pass


class TestDatasetMatching(BaseMultiSubjectTestCase):
    pass


class TestDicomTagMatch(BaseTestCase):

    IMAGE_TYPE_TAG = ('0008', '0008')
    GRE_PATTERN = 'gre_field_mapping_3mm.*'
    PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']
    MAG_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']
    DICOM_MATCH = [
        DatasetMatch('gre_phase', dicom_format, GRE_PATTERN,
                     dicom_tags={IMAGE_TYPE_TAG: PHASE_IMAGE_TYPE},
                     is_regex=True),
        DatasetMatch('gre_mag', dicom_format, GRE_PATTERN,
                     dicom_tags={IMAGE_TYPE_TAG: MAG_IMAGE_TYPE},
                     is_regex=True)]

    def test_dicom_match(self):
        study = self.create_study(
            TestMatchStudy, 'test_dicom',
            inputs=self.DICOM_MATCH)
        phase = study.data('gre_phase')[0]
        mag = study.data('gre_mag')[0]
        self.assertEqual(phase.name, 'gre_field_mapping_3mm_phase')
        self.assertEqual(mag.name, 'gre_field_mapping_3mm_mag')

    def test_order_match(self):
        study = self.create_study(
            TestMatchStudy, 'test_dicom',
            inputs=[
                DatasetMatch('gre_phase', dicom_format,
                             pattern=self.GRE_PATTERN, order=1,
                             is_regex=True),
                DatasetMatch('gre_mag', dicom_format,
                             pattern=self.GRE_PATTERN, order=0,
                             is_regex=True)])
        phase = study.data('gre_phase')[0]
        mag = study.data('gre_mag')[0]
        self.assertEqual(phase.name, 'gre_field_mapping_3mm_phase')
        self.assertEqual(mag.name, 'gre_field_mapping_3mm_mag')
