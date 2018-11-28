from arcana.utils.testing import BaseTestCase, BaseMultiSubjectTestCase
from arcana.study.base import Study, StudyMetaClass
from arcana.data import FilesetSpec, FilesetSelector
from banana.file_format import (
    dicom_format)


class TestMatchStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('gre_phase', dicom_format),
        FilesetSpec('gre_mag', dicom_format)]

    def dummy_pipeline1(self):
        pass

    def dummy_pipeline2(self):
        pass


class TestFilesetSelectoring(BaseMultiSubjectTestCase):
    pass


class TestDicomTagMatch(BaseTestCase):

    IMAGE_TYPE_TAG = ('0008', '0008')
    GRE_PATTERN = 'gre_field_mapping_3mm.*'
    PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']
    MAG_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']
    DICOM_MATCH = [
        FilesetSelector('gre_phase', dicom_format, GRE_PATTERN,
                     dicom_tags={IMAGE_TYPE_TAG: PHASE_IMAGE_TYPE},
                     is_regex=True),
        FilesetSelector('gre_mag', dicom_format, GRE_PATTERN,
                     dicom_tags={IMAGE_TYPE_TAG: MAG_IMAGE_TYPE},
                     is_regex=True)]

    def test_dicom_match(self):
        study = self.create_study(
            TestMatchStudy, 'test_dicom',
            inputs=self.DICOM_MATCH)
        phase = list(study.data('gre_phase'))[0]
        mag = list(study.data('gre_mag'))[0]
        self.assertEqual(phase.name, 'gre_field_mapping_3mm_phase')
        self.assertEqual(mag.name, 'gre_field_mapping_3mm_mag')

    def test_order_match(self):
        study = self.create_study(
            TestMatchStudy, 'test_dicom',
            inputs=[
                FilesetSelector('gre_phase', dicom_format,
                             pattern=self.GRE_PATTERN, order=1,
                             is_regex=True),
                FilesetSelector('gre_mag', dicom_format,
                             pattern=self.GRE_PATTERN, order=0,
                             is_regex=True)])
        phase = list(study.data('gre_phase'))[0]
        mag = list(study.data('gre_mag'))[0]
        self.assertEqual(phase.name, 'gre_field_mapping_3mm_phase')
        self.assertEqual(mag.name, 'gre_field_mapping_3mm_mag')
