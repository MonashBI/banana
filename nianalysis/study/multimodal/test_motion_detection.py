from nianalysis.study.mri.motion_detection_mixin_new import (
    create_motion_detection_class)

ref = 'ref'
t1s = ['ute']  # ['t1']
t2s = ['t2']
umap_ref = 'ute'
epis = []  # ['epi']
dmris = []  # [['dwi_main', '0'], ['dwi_opposite', '-1'], ['dwi_ref', '1']]
umap = ['umap']


class A(object):
    pass


MotionDetection, inputs = create_motion_detection_class(
    'MotionDetection', ref, 't1', t1s=t1s, t2s=t2s, epis=epis, dmris=dmris,
    umap_ref=umap_ref, umaps=umap)

MotionDetection.__module__ = A.__module__
