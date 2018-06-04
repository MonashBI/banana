from nianalysis.study.mri.motion_detection_mixin import (
    create_motion_detection_class)
import os.path
import errno
from arcana.repository.local import LocalRepository
from arcana.repository.xnat import XnatRepository


def run_md(input_dir, dynamic=False, xnat_id=None):

    if xnat_id is not None:
        repository = XnatRepository(cache_dir=input_dir+'/motion_detection_cache')
        work_dir = input_dir
        project_id = xnat_id.split('_')[0]
        sub_id = xnat_id.split('_')[0]+'_'+xnat_id.split('_')[1]
        session_id = xnat_id.split('_')[2]
    else:
        repository = LocalRepository(input_dir)
        work_dir = os.path.join(input_dir, 'motion_detection_cache')
        project_id = 'work_dir'
        sub_id = 'work_sub_dir'
        session_id = 'work_session_dir'

    ref = 'Head_t1_mprage_sag_p2_iso'
    t1s = ['Head_MAN_SHIM_T1_fl3d_sag_p3_iso_magnitude']
    cls, inputs = create_motion_detection_class(
        'motion_mixin', ref, ref_type='t1', t1s=t1s, dynamic=dynamic)
    print inputs
    WORK_PATH = work_dir
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    study = cls(name='motion_detection', project_id=project_id,
                repository=repository, inputs=inputs)
    study.gather_outputs_pipeline().run(
        subject_ids=[sub_id],
        visit_ids=[session_id], work_dir=WORK_PATH)


input_dir = '/Volumes/ELEMENTS/test_mc_mixin_folder/'
xnat_id = 'MMH008_CON012_MRPT01'
dynamic_md = False

run_md(input_dir, dynamic=dynamic_md, xnat_id=xnat_id)
