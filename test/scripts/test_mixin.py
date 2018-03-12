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
import os.path
import errno
from nianalysis.archive.local import LocalArchive
from nianalysis.archive.xnat import XNATArchive


list_t1 = ['t1_1_dicom', 't1_2_dicom']
list_t2 = ['t2_1_dicom', 't2_2_dicom']
list_epi = ['epi_1_dicom', 'asl_dicom']
list_dwi = [['dwi_main_dicom', '0'], ['dwi_opp_dicom', '-1'],
            ['dwi_ref_dicom', '1']]
list_utes = ['ute_dicom']

cls, inputs = create_motion_detection_class(
    'test_mixin', 'Head_t1_mprage_sag_p2_iso', 't1', t1s=list_t1, t2s=list_t2, dmris=list_dwi,
    epis=list_epi, utes=list_utes, umaps=['umap_dicom'])

WORK_PATH = os.path.join('/Users', 'francescosforazzini', 'Desktop',
                         'test_mc_mixin')
try:
    os.makedirs(WORK_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

study = cls(
    name='test_mc_mixin',
    project_id='MMH008', archive=LocalArchive(
        '/Users/francescosforazzini/Desktop/test_mc_mixin_cache'),

    inputs=inputs)
study.plot_mean_displacement_pipeline().run(
    subject_ids=['MMH008_{}'.format(i) for i in ['CON012']],
    visit_ids=['MRPT01'], work_dir=WORK_PATH)

print 'Done!'
