#!/usr/bin/env python
from arcana.dataset import DatasetMatch
from nianalysis.study.mri.phantom import PhantomStudy
from arcana.archive.xnat import XNATArchive
from nianalysis.data_format import dicom_format

qc_study = PhantomStudy(
    name='qc',
    project_id='INSTRUMENT',
    archive=XNATArchive(server='https://mbi-xnat.erc.monash.edu.au'),
    inputs=['t1_32ch_saline': Dataset('t1_mprage_trans_p2_iso_0.9_32CH',
                                      dicom_format),
            DatasetMatch('t2_32ch_saline', dicom_format, 't2_spc_tra_iso_32CH'),
            DatasetMatch('diffusion_32ch_saline', dicom_format, 'ep2d_diff_mddw_12_p2_32CH')})


qc_study.epi_32ch_qc_metrics_pipeline().run(visit_ids=['20170724', '20170807'],
                                            reprocess='all')
