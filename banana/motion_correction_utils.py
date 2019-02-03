import os.path
import pydicom
import glob
import shutil
import errno
import subprocess as sp
from banana.interfaces.custom.dicom import DicomHeaderInfoExtraction
import numpy as np
import re
import datetime as dt


PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']


# def xnat_motion_detection(xnat_id):
#
#     avail_scans = xnat_ls(xnat_id, datatype='scan')
#     print(avail_scans)
#
#     return avail_scans


def local_motion_detection(input_dir, pet_dir=None, pet_recon=None,
                           struct2align=None):

    scan_description = []
    dcm_files = sorted(glob.glob(input_dir+'/*.dcm'))
    if not dcm_files:
        dcm_files = sorted(glob.glob(input_dir+'/*.IMA'))
    else:
        dcm = True
    if not dcm_files:
        scan_description = [f for f in os.listdir(input_dir) if (not
                            f.startswith('.') and os.path.isdir(input_dir+f)
                            and 'motion_correction_results' not in f)]
        dcm = False
    else:
        dcm = True
    if not dcm_files and not scan_description:
        raise Exception('No DICOM files or folders found in {}'
                        .format(input_dir))
    try:
        os.mkdir(input_dir+'/work_dir')
        os.mkdir(input_dir+'/work_dir/work_sub_dir')
        os.mkdir(input_dir+'/work_dir/work_sub_dir/work_session_dir')
        working_dir = input_dir+'/work_dir/work_sub_dir/work_session_dir/'
        copy = True
    except OSError as e:
        if e.errno == errno.EEXIST:
            print ('Detected existing working directory. Assuming that a '
                   'previous process failed. Trying to restart it.')
            working_dir = input_dir+'/work_dir/work_sub_dir/work_session_dir/'
            copy = False
    if dcm:
        hdr = pydicom.read_file(dcm_files[0])
        name_scan = (
            str(hdr.SeriesNumber).zfill(2)+'_'+hdr.SeriesDescription)
        name_scan = name_scan.replace(" ", "_")
        scan_description = [name_scan]
        files = []
        for i, im in enumerate(dcm_files):
            hdr = pydicom.read_file(im)
            name_scan = (
                str(hdr.SeriesNumber).zfill(2)+'_'+hdr.SeriesDescription)
            name_scan = name_scan.replace(" ", "_")
            if name_scan in scan_description[-1]:
                files.append(im)
            elif name_scan not in scan_description[-1] and copy:
                if (os.path.isdir(
                        working_dir+scan_description[-1]) is False):
                    os.mkdir(working_dir+scan_description[-1])
                    for f in files:
                        shutil.copy(f, working_dir+scan_description[-1])
                files = [im]
                scan_description.append(name_scan)
            elif name_scan not in scan_description[-1] and not copy:
                files = [im]
                scan_description.append(name_scan)
            if i == len(dcm_files)-1 and copy:
                if (os.path.isdir(working_dir+scan_description[-1]) is
                        False):
                    os.mkdir(working_dir+scan_description[-1])
                    for f in files:
                        shutil.copy(f, working_dir+scan_description[-1])
    elif not dcm and copy:
        for s in scan_description:
            shutil.copytree(input_dir+s, working_dir+'/'+s)
        if pet_dir is not None:
            shutil.copytree(pet_dir, working_dir+'/pet_data_dir')
        if pet_recon is not None:
            shutil.copytree(pet_recon, working_dir+'/pet_data_reconstructed')
        if struct2align is not None:
            shutil.copy2(struct2align, working_dir+'/')

    phase_image_type, no_dicom = check_image_type(input_dir, scan_description)
    if no_dicom:
        print(('No DICOM files could be found in the following folders '
               'For this reason they will be removed from the analysis.\n{}'
               .format('\n'.join(x for x in no_dicom))))
        scan_description = [x for x in scan_description
                            if x not in no_dicom]
    if phase_image_type:
        print(('The following scans were found to be phase image '
               'For this reason they will be removed from the analysis.\n{}'
               .format('\n'.join(x for x in phase_image_type))))
        scan_description = [x for x in scan_description
                            if x not in phase_image_type]
    same_start_time = check_image_start_time(input_dir, scan_description)
    if same_start_time:
        print(('The following scans were found to have the same start time '
               'as other scans provided. For this reason they will be removed'
               ' from the analysis.\n{}'
               .format('\n'.join(x for x in same_start_time))))
        scan_description = [x for x in scan_description
                            if x not in same_start_time]

    return scan_description


def inputs_generation(scan_description, input_dir, siemens=False):

    ref = None
    umaps = None
    scan_description = sorted(scan_description)

    for i, scan in enumerate(scan_description, start=1):
        if i == 1:
            print('Available scans: ')
        print('{0} {1}'.format(i, scan))
    correct_ref = False
    while not correct_ref:
        ref = input("Please select the reference scan: ").split()
        if not ref:
            print ('A reference image must be provided!')
        elif ref and len(ref) > 1:
            print(('Only one reference can be provided, you selected {}.'
                   .format(len(ref))))
        else:
            correct_ref = True

    ref = scan_description[int(ref[0])-1]
    correct_ref_type = False
    while not correct_ref_type:
        ref_type = input(
            "Please enter the reference type ('t1' or 't2'): ").split()[0]
        if ref_type != 't1' and ref_type != 't2':
            print(('{} is not a recognized ref_type!The available '
                   'ref_types are t1 or t2.'.format(ref_type)))
        else:
            correct_ref_type = True
    t1s = input("Please select the T1 weighted scans: ").split()
    epis = input("Please select the EPI scans: ").split()
    dwi = input("Please select the DWI scans (including both b0 and"
                " main diffusion images): ").split()
    dwi = [scan_description[int(i)-1] for i in dwi]
    if dwi:
        dwis, unused_b0 = dwi_type_assignment(input_dir, dwi)
        if len(dwis) < len(dwi):
            raise Exception('The following DWI scan cannot be recognized as b0'
                            'or main diffusion scan. Please remove it from the'
                            'input directory or assign it to others:\n{}'
                            .format(' '.join(x for x in dwi if x not in
                                             dwis)))
        else:
            print ('The DWI images provided were assigned to the following '
                   'types: \n')
            for dwi in dwis:
                if dwi[-1] == '0':
                    print(('main diffusion image with multiple directions: '
                           '{}'.format(dwi[0])))
                elif dwi[-1] == '1':
                    print(('b0 image with same phase encoding direction '
                           'respect to the main dwi: {}'.format(dwi[0])))
                elif dwi[-1] == '-1':
                    print(('b0 image with opposite phase encoding direction '
                           'respect to the main dwi: {}'.format(dwi[0])))
            print('\n')
    if not siemens:
        utes = input("Please select the UTE scans: ").split()
        utes = [scan_description[int(i)-1] for i in utes]
        if utes:
            umaps = input("Please select the umap: ").split()
            umaps = [scan_description[int(i)-1] for i in umaps]

    t2s = input("Please select the all the other scans the do not belog to"
                " any of the previous classes: ").split()
    t1s = [scan_description[int(i)-1] for i in t1s]
    t2s = [scan_description[int(i)-1] for i in t2s]
    epis = [scan_description[int(i)-1] for i in epis]
    if unused_b0:
        print(('The following b0 images with different phase encoding '
               'direction respect to the main diffusion has been found. They '
               'will be treated as T2w:\n{}'
               .format('\n'.join(x for x in unused_b0))))
        t2s = t2s+unused_b0

    if siemens:
        return ref, ref_type, t1s, epis, t2s, dwis
    else:
        return ref, ref_type, t1s, epis, t2s, dwis, utes, umaps


def guess_scan_type(scans, input_dir):

    ref = None
    ref_type = None
    t1s = []
    t2s = []
    epis = []
    dwi_scans = []
    res_t1 = []
    res_t2 = []

    for scan in scans:
        sequence_name = None
        dcm_files = sorted(glob.glob(input_dir+'/'+scan+'/*.dcm'))
        if not dcm_files:
            dcm_files = sorted(glob.glob(input_dir+'/'+scan+'/*.IMA'))
        if not dcm_files:
            continue
        dicom = dcm_files[0]
        hd = pydicom.read_file(dicom)
        with open(dcm_files[0], 'rb') as f:
            for line in f:
                try:
                    line = line[:-1].decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if 'tSequenceFileName' in line:
                    sequence_name = line.strip().split('\\')[-1].split('"')[0]
                    break

        if sequence_name is not None:
            if (('tfl' in sequence_name or
                    re.match('.*(ute|UTE).*', sequence_name)) or
                    (re.match('.*(t1|T1).*', scan) or
                     re.match('.*(ute|UTE).*', scan))):
                t1s.append(scan)
                res_t1.append([scan, float(hd.PixelSpacing[0])])
            elif 'bold' in sequence_name or 'asl' in sequence_name:
                epis.append(scan)
            elif 'diff' in sequence_name:
                dwi_scans.append(scan)
            else:
                t2s.append(scan)
                if 'gre' not in sequence_name:
                    res_t2.append([scan, float(hd.PixelSpacing[0])])
    dwis, unused_b0 = dwi_type_assignment(input_dir, dwi_scans)
    if unused_b0:
        print(('The following b0 images have different phase encoding '
               'direction respect to the main diffusion and/or the ped '
               'information could not be found in the image header. They '
               'will be treated as T2w:\n{}'
               .format('\n'.join(x for x in unused_b0))))
        t2s = t2s+unused_b0
    if res_t2:
        ref = [x[0] for x in res_t2 if x[1] <= 1]
        if ref:
            if len(ref) > 1:
                min_res = np.min([x[1] for x in res_t2])
                ref = [x[0] for x in res_t2 if x[1] == min_res]
            ref_type = 't2'
            ref = ref[0]
            t2s.remove(ref)
    if res_t1 and not ref:
        ref = [x[0] for x in res_t1 if x[1] <= 1]
        if ref:
            ref_type = 't1'
            ref = ref[0]
            t1s.remove(ref)
    if ref:
        print(('\nChosen reference image: {0} \nReference type: {1}\n'
               .format(ref, ref_type)))
        if t1s:
            print ('The following scans were identified as T1 weighted: \n')
            for t1 in t1s:
                print(('{}'.format(t1)))
            print('\n')
        if epis:
            print ('The following scans were identified as 4D BOLD or ASL '
                   'images:\n')
            for epi in epis:
                print(('{}'.format(epi)))
            print('\n')
        if dwis:
            print ('The following scans were identified as DWI: \n')
            for dwi in dwis:
                if dwi[-1] == '0':
                    print(('main diffusion image with multiple directions: '
                           '{}'.format(dwi[0])))
                elif dwi[-1] == '1':
                    print(('b0 image with same phase encoding direction '
                           'respect to the main dwi: {}'.format(dwi[0])))
                elif dwi[-1] == '-1':
                    print(('b0 image with opposite phase encoding direction '
                           'respect to the main dwi: {}'.format(dwi[0])))
            print('\n')
        if t2s:
            print ('The following scans were identified as not belonging to '
                   'any of the previous classes: \n')
            for t2 in t2s:
                print(('{}'.format(t2)))
            print('\n')

        assigned = [ref]+t1s+t2s+[x[0] for x in dwis]+epis
        not_assigned = [x for x in scans if x not in assigned]
        if not_assigned:
            print (
                'The following scans could not be assigned to any class.'
                ' If DWI images are present, this might be due to the '
                'inhability of the pipeline to find phase encoding information'
                'in the header. \n')
            for na in not_assigned:
                print(('{}'.format(na)))
            print('\n')

        check_guess = input(
            "Please type 'yes' if the grouping is correct, otherwise 'no'. "
            "If it is not correct then you will be prompted to manually group "
            "all the scans into the different classes: ").split()
        if check_guess[0] == 'yes':
            inputs = [ref, ref_type, t1s, epis, t2s, dwis]
        else:
            inputs = []
    else:
        print ('Reference image could not be identified from header '
               'information. You will be prompted to manually group all '
               'the scans into different classes.')
        inputs = []

    return inputs


def dwi_type_assignment(input_dir, dwi_images):

    main_dwi = []
    b0 = []
    dwis = []
    unused_b0 = []

    for dwi in dwi_images:
        cmd = 'mrinfo {0}'.format(input_dir+'/'+dwi)
        info = (sp.check_output(cmd, shell=True)).decode('utf-8')
        info = info.strip().split('\n')
        for line in info:
            if 'Dimensions:' in line:
                dim = line.split('Dimensions:')[-1].strip().split('x')
                break
        hd_extraction = DicomHeaderInfoExtraction()
        hd_extraction.inputs.dicom_folder = input_dir+'/'+dwi
        dcm_info = hd_extraction.run()

        if dcm_info.outputs.pe_angle and dcm_info.outputs.ped:
            if len(dim) == 4:
                main_dwi.append(
                    [dwi, np.trunc(float(dcm_info.outputs.pe_angle)),
                     dcm_info.outputs.ped])
            else:
                b0.append([
                    dwi, np.trunc(float(dcm_info.outputs.pe_angle)),
                    dcm_info.outputs.ped])
        else:
            print ('Could not find phase encoding information from the'
                   'dwi images header. Distortion correction will not '
                   'be performed.')
            if len(dim) == 4:
                main_dwi.append(
                    [dwi, '', ''])
            else:
                b0.append([
                    dwi, '-2', '-2'])

    for i in range(len(main_dwi)):
        if main_dwi[i][2]:
            dwis.append([main_dwi[i][0], '0'])
        ped_main = main_dwi[i][2]
        for j in range(len(b0)):
            ped_b0 = b0[j][2]
            if ped_b0 == ped_main:
                if main_dwi[i][1] == b0[j][1] and (j == i or j == i+1):
                    dwis.append([b0[j][0], '1'])
                elif main_dwi[i][1] != b0[j][1] and (j == i or j == i+1):
                    dwis.append([b0[j][0], '-1'])
            else:
                unused_b0.append(b0[j][0])
#         if not b0_found:
#             raise Exception(
#                 'The phase encoding direction between the main'
#                 'DWI and the provided b0 images is not the '
#                 'same! Please check.')

    return dwis, unused_b0


def check_image_type(input_dir, scans):

    toremove = []
    nodicom = []
    for scan in scans:
        dcm_file = None
        try:
            dcm_file = sorted(glob.glob(input_dir+'/'+scan+'/*.dcm'))[0]
        except IndexError:
            try:
                dcm_file = sorted(glob.glob(
                    input_dir+'/'+scan+'/*.IMA'))[0]
            except IndexError:
                nodicom.append(scan)
        if dcm_file is not None:
            try:
                hd = pydicom.read_file(dcm_file)
                im_type = hd['0008', '0008'].value
                if im_type == PHASE_IMAGE_TYPE:
                    toremove.append(scan)
            except:
                print(('{} does not have the image type in the header. It will'
                       ' be removed from the analysis'.format(scan)))

    return toremove, nodicom


def check_image_start_time(input_dir, scans):

    start_times = []
    toremove = []
    for scan in scans:
        try:
            scan_number = scan.split('-')[0].zfill(3)
            hd_extraction = DicomHeaderInfoExtraction()
            hd_extraction.inputs.dicom_folder = input_dir+'/'+scan
            dcm_info = hd_extraction.run()
            start_times.append([dcm_info.outputs.start_time, scan_number,
                                scan])
        except:
            print(('This folder {} seems to not contain DICOM files. It will '
                   'be ingnored.'.format(scan)))
    start_times = sorted(start_times)
    for i in range(1, len(start_times)):
        diff = ((dt.datetime.strptime(start_times[i][0], '%H%M%S.%f') -
                dt.datetime.strptime(start_times[i-1][0], '%H%M%S.%f'))
                .total_seconds())
        if diff < 5:
            toremove.append(start_times[i][-1])

    return toremove


def md_cleanup(input_dir, work_dir, project_id, sub_id, session_id):

    shutil.move(os.path.join(
        input_dir, project_id, sub_id, session_id,
        'motion_detection_motion_detection_output'), input_dir)
    os.rename(
        os.path.join(input_dir, 'motion_detection_motion_detection_output'),
        os.path.join(input_dir, 'motion_detection_output'))
    shutil.rmtree(work_dir)
    shutil.rmtree(os.path.join(input_dir, project_id))
