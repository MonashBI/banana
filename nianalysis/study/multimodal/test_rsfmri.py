from nianalysis.study.mri.functional.fmri_new import create_fmri_study_class

t1 = 't1'  # ['t1']
epis = ['epi_1']  # ['epi']
fm_mag = 'field_map_mag'
fm_phase = 'field_map_phase'
train_set = 'rsfPET_training_set'


class A(object):
    pass


fMRI, inputs, output_file = create_fmri_study_class(
    'fMRI', t1, epis, fm_mag, fm_phase, training_set=train_set)

fMRI.__module__ = A.__module__
