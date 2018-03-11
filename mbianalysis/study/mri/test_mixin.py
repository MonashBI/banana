from nianalysis.study.base import set_specs
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from mbianalysis.study.mri.epi import CoregisteredEPIStudy
from .structural.diffusion_coreg import (
    CoregisteredDiffusionStudy,
    CoregisteredDiffusionReferenceStudy, CoregisteredDiffusionOppositeStudy)
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, text_format,
    png_format, dicom_format)
from nianalysis.dataset import Dataset
from .structural.t1 import CoregisteredT1Study
from .structural.t2 import CoregisteredT2Study
from .motion_detection_mixin import MotionReferenceT1Study, MotionReferenceT2Study, MotionDetectionMixin 

def create_motion_detection_class(name, reference, ref_type, t1s=None,
                                  t2s=None, dmris=None, epis=None):

    inputs = {}
    if ref_type == 't1':
        ref_study = MotionReferenceT1Study
    elif ref_type == 't2':
        ref_study = MotionReferenceT2Study

    study_specs = [SubStudySpec('ref', ref_study)]
    ref_spec = {'ref_preproc': 'ref_preproc',
                'ref_masked': 'ref_brain',
                'ref_brain_mask': 'ref_brain_mask'}
    inputs['ref_primary'] = Dataset(reference, dicom_format)
    
    if t1s is not None:
        for i, t1_scan in enumerate(t1s):
            study_specs.extend(
                SubStudySpec('t1_{}_t1'.format(i), CoregisteredT1Study,
                             ref_spec))
            inputs['t1_{}_t1'.format(i)] = Dataset(t1_scan, dicom_format)
    if t2s is not None:
        for i, t2_scan in enumerate(t2s):
            study_specs.extend(
                SubStudySpec('t2_{}_t2'.format(i), CoregisteredT2Study,
                             ref_spec))
            inputs['t2_{}_t2'.format(i)] = Dataset(t2_scan, dicom_format)
    if epis is not None:
        for i, epi_scan in enumerate(epis):
            study_specs.extend(
                SubStudySpec('epi_{}_epi'.format(i), CoregisteredEPIStudy,
                             ref_spec.update({'ref_wm_seg': 'ref_wmseg'})))
            inputs['epi_{}_epi'.format(i)] = Dataset(epi_scan, dicom_format)
    if dmris is not None:
        dmris_main = [x for x in dmris if x[-1]=='0']
        dmris_ref = [x for x in dmris if x[-1]=='1']
        dmris_opposite = [x for x in dmris if x[-1]=='-1']
        if dmris_main and not dmris_opposite:
            raise Exception('If you provide one main Diffusion image you '
                            'have also to provide an opposite ped image.')
        if dmris_main and dmris_opposite and (len(dmris_main) == len(dmris_opposite)):
            for i, dmris_main_scan in enumerate(dmris_main):
                study_specs.extend(
                    SubStudySpec('dwi_{}_dwi_main'.format(i), CoregisteredDiffusionStudy,
                                 ref_spec))
                inputs['dwi_{}_dwi_main'.format(i)] = Dataset(
                    dmris_main_scan, dicom_format)
                inputs['dwi_{}_dwi_main_ref'.format(i)] = Dataset(
                    dmris_opposite[i], dicom_format)
            if not dmris_ref:
                study_specs.extend(
                    SubStudySpec(
                        'dwi_{}_dwi_opposite'.format(i),
                        CoregisteredDiffusionOppositeStudy, ref_spec))
                inputs['dwi_{}_dwi_opposite_to_correct'.format(i)] = Dataset(
                    dmris_opposite[i], dicom_format)
                inputs['dwi_{}_dwi_opposite_ref'.format(i)] = Dataset(
                    dmris_main_scan, dicom_format)
        elif dmris_main and dmris_opposite and (len(dmris_main) != len(dmris_opposite)):
            for i, dmris_main_scan in enumerate(dmris_main):
                study_specs.extend(
                    SubStudySpec('dwi_{}_dwi_main'.format(i), CoregisteredDiffusionStudy,
                                 ref_spec))
                inputs['dwi_{}_dwi_main'.format(i)] = Dataset(
                    dmris_main_scan, dicom_format)
                inputs['dwi_{}_dwi_main_ref'.format(i)] = Dataset(
                    dmris_opposite[0], dicom_format)
            if not dmris_ref:
                for i, dmris_opp_scan in enumerate(dmris_opposite):
                    study_specs.extend(
                        SubStudySpec(
                            'dwi_{}_dwi_opposite'.format(i),
                            CoregisteredDiffusionOppositeStudy, ref_spec))
                    inputs['dwi_{}_dwi_opposite_to_correct'.format(i)] = Dataset(
                        dmris_opp_scan, dicom_format)
                    inputs['dwi_{}_dwi_opposite_ref'.format(i)] = Dataset(
                        dmris_main[0], dicom_format)
        if dmris_ref and (len(dmris_ref) == len(dmris_opposite)):
            for i, dmris_ref_scan in enumerate(dmris_ref):
                study_specs.extend(
                    SubStudySpec('dwi_{}_toref'.format(i),
                                 CoregisteredDiffusionReferenceStudy,
                                 ref_spec))
                study_specs.extend(
                    SubStudySpec(
                        'dwi_{}_dwi_opposite'.format(i),
                        CoregisteredDiffusionOppositeStudy, ref_spec))
                inputs['dwi_{}_dwi_opposite_to_correct'.format(i)] = Dataset(
                    dmris_opposite[i], dicom_format)
                inputs['dwi_{}_dwi_opposite_ref'.format(i)] = Dataset(
                    dmris_ref_scan, dicom_format)
                inputs['dwi_{}_toref_dwi2ref_to_correct'.format(i)] = Dataset(
                    dmris_ref_scan, dicom_format)
                inputs['dwi_{}_dwi_dwi2ref_ref'.format(i)] = Dataset(
                    dmris_opposite[i], dicom_format)
        elif dmris_ref and (len(dmris_ref) != len(dmris_opposite)):
            for i, dmris_ref_scan in enumerate(dmris_ref):
                study_specs.extend(
                    SubStudySpec('dwi_{}_toref'.format(i),
                                 CoregisteredDiffusionReferenceStudy,
                                 ref_spec))
                inputs['dwi_{}_toref_dwi2ref_to_correct'.format(i)] = Dataset(
                    dmris_ref_scan, dicom_format)
                inputs['dwi_{}_dwi_dwi2ref_ref'.format(i)] = Dataset(
                    dmris_opposite[0], dicom_format)
            for i, dmris_opp_scan in enumerate(dmris_opposite):
                study_specs.extend(
                    SubStudySpec(
                        'dwi_{}_dwi_opposite'.format(i),
                        CoregisteredDiffusionOppositeStudy, ref_spec))
                inputs['dwi_{}_dwi_opposite_to_correct'.format(i)] = Dataset(
                    dmris_opp_scan, dicom_format)
                inputs['dwi_{}_dwi_opposite_ref'.format(i)] = Dataset(
                    dmris_ref[0], dicom_format)
    dct = {}
    dct['_sub_study_specs'] = set_specs(study_specs)
    dct['_data_specs'] = {}
    return MultiStudyMetaClass(name, [MotionDetectionMixin], dct), inputs

list_t1 = ['t1_1_dicom', 't1_2_dicom']
list_t2 = ['t2_1_dicom', 't2_2_dicom']
list_epi = ['epi_1_dicom', 'asl_dicom']
list_dwi = [['dwi_main_dicom', '0'], ['dwi_opp_dicom', '-1'], ['dwi_ref_dicom', '1']]

create_motion_detection_class('test_mixin', 't1_dicom' , 't1', t1s=list_t1,
                                  t2s=list_t2, dmris=list_dwi, epis=list_epi)