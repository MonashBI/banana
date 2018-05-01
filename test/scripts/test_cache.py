from mbianalysis.data_format import dicom_format
from nianalysis.dataset import DatasetMatch
from nianalysis.archive.xnat import XNATArchive


archive = XNATArchive()

archive.cache(
    'MRH032',
    [Dataset('t1_mprage_sag_p2_iso_1mm', format=dicom_format),
     Dataset('t2_tra_tse_320_4mm', format=dicom_format)],
    subject_ids=['MRH032_{:03}'.format(i) for i in range(1, 20)],
    visit_ids=['MR01', 'MR03'])
