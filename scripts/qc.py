#!/usr/bin/env python
from nianalysis.dataset import Dataset
from nianalysis.study.mri.phantom import PhantomStudy
from nianalysis.archive.xnat import XNATArchive
from nianalysis.data_formats import dicom_format

study = PhantomStudy(
    name='phantom',
    project_id='INSTRUMENT', archive=XNATArchive(),
    inputs={'qc': Dataset('32 CH T1_mprage_trans_p2_iso_0.9',
                          dicom_format)})
study.qc_metrics_pipeline().run(visit_ids=['20170724', '20170807'])
