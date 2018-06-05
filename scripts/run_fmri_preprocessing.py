from nianalysis.study.mri.functional.fmri_new import create_fmri_study_class
from arcana.repository.xnat import XnatRepository
from arcana.runner.linear import LinearRunner
import os.path
import errno
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--project_id', '-p', type=str, required=True,
                        help='Project ID on XNAT.')
    parser.add_argument('--working_dir', '-wd', type=str, required=True,
                        help='Path to local directory to cache XNAT data in '
                        'and to carry out all the processing.')
    parser.add_argument('--hires_structural', '-struct', type=str,
                        required=True, help='High resolution structural image '
                        'to used to improve the registation between fMRI and '
                        'MNI template (usually is a T1 weigthed image).')
    parser.add_argument('--fmri', type=str, required=True,
                        help='Regular expression to match the name of the fMRI'
                        ' images on XNAT to be pre-processed. If more than one'
                        ' image match this expression, please provide the '
                        '--fmri_order as well. Also, in case of multiple fmri '
                        ' images, the same field map images (if provided) will'
                        ' be used to perform B0 unwarping.')
    parser.add_argument('--fmri_order', type=int, required=True,
                        help='If more than one fmri image is going to match '
                        'the --fmri regular expression provided, you can '
                        'specify which one to use. For example, if there are 5'
                        ' matches and you specify order=3 and only the first 3'
                        ' images will be processed PER SUBJECT. Please be '
                        'aware that if one subject as a number of fmri less '
                        'than the order, this will cause an error. In this '
                        'case you may want to process that subject '
                        'independently. Default is the first match.',
                        default=0)
    parser.add_argument('--session_ids', '-s', type=str,
                        help='Session ID on XNAT. Default is all of the '
                        'sessions found in the XNAT project.')
    parser.add_argument('--subject_ids', '-sub', type=str,
                        help='Subject ID on XNAT. Default is all the '
                        'subjects found in all the session specified.')
    parser.add_argument('--xnat_server', '-server', type=str,
                        help='URI of XNAT server to connect to. Default is '
                        'https://mbi-xnat.erc.monash.edu.au',
                        default='https://mbi-xnat.erc.monash.edu.au')
    parser.add_argument('--xnat_username', '-user', type=str,
                        help='Username with which to connect to XNAT with. '
                        'This can be skiped if it has already been saved in '
                        'the .netrc in your home directory', default=None)
    parser.add_argument('--xnat_password', '-password', type=str,
                        help='Password to connect to XNAt with. '
                        'This can be skiped if it has already been saved in '
                        'the .netrc in your home directory', default=None)
    parser.add_argument('--field_map_mag', '-mag', type=str,
                        help='Magnitude field map image used to correct for B0'
                        ' inhomogeneity. For the correction to take place, '
                        'this must be provided together with field_map_phase.',
                        default=None)
    parser.add_argument('--field_map_phase', '-phase', type=str,
                        help='Phase field map image used to correct for B0'
                        ' inhomogeneity. For the correction to take place, '
                        'this must be provided together with field_map_mag.'
                        ' N.B. right now, this pipeline assumes that this '
                        'phase image is the output of a SIEMENS scanner. It '
                        'does not support other vendors.',
                        default=None)
    parser.add_argument('--fix_training_set', '-ts', type=str,
                        help='Trainging set for FSL-FIX (.RData format). If '
                        'provided, fix classification and regression of the '
                        'noisy component will be performed and the final image'
                        ' will be fully pre-processed. Otherwise, the '
                        'pipeline will generate only the MELODIC L1 results.',
                        default=None)
    args = parser.parse_args()

    fMRI, inputs, output_files = create_fmri_study_class(
            'fMRI', args.hires_structural, args.fmri, args.fmri_order,
            fm_mag=args.field_map_mag, fm_phase=args.field_map_phase,
            training_set=args.fix_training_set)

    CACHE_PATH = os.path.join(args.working_dir, 'xnat_cache')
    WORK_PATH = os.path.join(args.working_dir, 'work_dir')
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(CACHE_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    sub_ids = args.subject_ids
    session_ids = args.session_ids
    repository = XnatRepository(
        server=args.xnat_server, project_id=args.project_id,
        user=args.xnat_username, password=args.xnat_password,
        cache_dir=CACHE_PATH)

    study = fMRI(name='fMRI_preprocessing', runner=LinearRunner(WORK_PATH),
                 repository=repository, inputs=inputs, subject_ids=[sub_ids],
                 visit_ids=[session_ids])
    study.data(output_files)

print('Done!')
