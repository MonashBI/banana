from banana.file_format import dicom_format
from arcana.data import FilesetSelector
from arcana.repository.xnat import XnatRepo


repository = XnatRepo()

repository.cache(
    'MRH032',
    [Fileset('t1_mprage_sag_p2_iso_1mm', format=dicom_format),
     Fileset('t2_tra_tse_320_4mm', format=dicom_format)],
    subject_ids=['MRH032_{:03}'.format(i) for i in range(1, 20)],
    visit_ids=['MR01', 'MR03'])
