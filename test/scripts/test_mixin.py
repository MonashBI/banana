from nianalysis.study.base import set_specs
from nianalysis.study.multi import (SubStudySpec, MultiStudyMetaClass)
from mbianalysis.study.mri.epi import CoregisteredEPIStudy
from mbianalysis.study.mri.structural.diffusion_coreg import (
    CoregisteredDiffusionStudy,
    CoregisteredDiffusionReferenceStudy, CoregisteredDiffusionOppositeStudy)
from nianalysis.data_formats import dicom_format
from nianalysis.dataset import Dataset
from mbianalysis.study.mri.structural.t1 import CoregisteredT1Study
from mbianalysis.study.mri.structural.t2 import CoregisteredT2Study
from mbianalysis.study.mri.motion_detection_mixin import (
    create_motion_detection_class)


list_t1 = ['t1_1_dicom', 't1_2_dicom']
list_t2 = ['t2_1_dicom', 't2_2_dicom']
list_epi = ['epi_1_dicom', 'asl_dicom']
list_dwi = [['dwi_main_dicom', '0'], ['dwi_opp_dicom', '-1'],
            ['dwi_ref_dicom', '1']]
list_utes = ['ute_dicom']

cls, inputs = create_motion_detection_class(
    'test_mixin', 't1_dicom', 't1', t1s=list_t1, t2s=list_t2, dmris=list_dwi,
    epis=list_epi, utes=list_utes, umaps=['umap_dicom'])

print(cls)
print(inputs)
