from nianalysis.study.multimodal.mrpet import (
    create_motion_detection_class)

ref = 'ref'
t1s = ['ute']  # ['t1']
t2s = ['t2']
umap_ref = 'ute'
epis = []  # ['epi']
dmris = []  # [['dwi_main', '0'], ['dwi_opposite', '-1'], ['dwi_ref', '1']]
umap = ['umap']
pet_data_dir = 'pet_data_dir'


class A(object):
    pass


MotionCorrection, inputs = create_motion_detection_class(
    'MotionCorrection', ref, 't1', t1s=t1s, t2s=t2s, epis=epis, dmris=dmris,
    umap_ref=umap_ref, umaps=umap, pet_data_dir=pet_data_dir)

MotionCorrection.__module__ = A.__module__
