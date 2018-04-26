from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec)
import nibabel as nib
import numpy as np
from nipype.utils.filemanip import split_filename
import os
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import subprocess as sp
from nipype.interfaces.base.traits_extension import Directory
import shutil
import glob
import pydicom
from nipype.interfaces import fsl


list_mode_framing_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'resources', 'C_C++',
                 'ListModeFraming'))
interfile_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'resources', 'pet',
                 'biograph_mmr_short_int.hs'))


class PETdrInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='4D input for the dual regression',
                  mandatory=True)
    regression_map = File(exists=True, desc='3D map to use for the spatial '
                          'regression (first step of the dr)', mandatory=True)
    threshold = traits.Float(desc='Threshold to be applied to the abs(reg_map)'
                             ' before regression (default zero)', default=0.0)
    binarize = traits.Bool(desc='If True, all the voxels greater than '
                           'threshold will be set to 1 (default False)',
                           default=False)


class PETdrOutputSpec(TraitedSpec):

    spatial_map = File(
        exists=True, desc='Nifti file containing result for the temporal '
        'regression')
    timecourse = File(
        exists=True, desc='Png file containing result for the spatial '
        'regression')


class PETdr(BaseInterface):

    input_spec = PETdrInputSpec
    output_spec = PETdrOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.volume
        mapname = self.inputs.regression_map
        th = self.inputs.threshold
        binarize = self.inputs.binarize
        _, base, _ = split_filename(fname)
        _, base_map, _ = split_filename(mapname)

        img = nib.load(fname)
        data = np.array(img.get_data())
        spatial_regressor = nib.load(mapname)
        spatial_regressor = np.array(spatial_regressor.get_data())

        n_voxels = data.shape[0]*data.shape[1]*data.shape[2]
        ts = data.reshape(n_voxels, data.shape[3])
        mask = spatial_regressor.reshape(n_voxels, 1)
        if th and not binarize:
            mask[np.abs(mask) < th] = 0
            base = base+'_th_{}'.format(str(th))
        elif th and binarize:
            mask[mask < th] = 0
            mask[mask >= th] = 1
            base = base+'_bin_th_{}'.format(str(th))
        timecourse = np.dot(ts.T, mask)
        sm = np.dot(ts, timecourse)
        mean = np.mean(sm)
        std = np.std(sm)
        sm_zscore = (sm-mean)/std

        im2save = nib.Nifti1Image(
            sm_zscore.reshape(spatial_regressor.shape), affine=img.affine)
        nib.save(
            im2save, '{0}_{1}_GLM_fit_zscore.nii.gz'.format(base, base_map))

        plot.plot(timecourse)
        plot.savefig('{0}_{1}_timecourse.png'.format(base, base_map))
        plot.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume
        th = self.inputs.threshold
        binarize = self.inputs.binarize
        mapname = self.inputs.regression_map

        _, base_map, _ = split_filename(mapname)
        _, base, _ = split_filename(fname)
        if th and not binarize:
            base = base+'_th_{}'.format(str(th))
        elif th and binarize:
            base = base+'_bin_th_{}'.format(str(th))

        outputs["spatial_map"] = os.path.abspath(
            '{0}_{1}_GLM_fit_zscore.nii.gz'.format(base, base_map))
        outputs["timecourse"] = os.path.abspath(
            '{0}_{1}_timecourse.png'.format(base, base_map))

        return outputs


class GlobalTrendRemovalInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='4D input file',
                  mandatory=True)


class GlobalTrendRemovalOutputSpec(TraitedSpec):

    detrended_file = File(
        exists=True, desc='4D file with the first temporal PCA component'
        ' removed')


class GlobalTrendRemoval(BaseInterface):

    input_spec = GlobalTrendRemovalInputSpec
    output_spec = GlobalTrendRemovalOutputSpec

    def _run_interface(self, runtime):

        fname = self.inputs.volume
        _, base, _ = split_filename(fname)

        img = nib.load(fname)
        data = np.array(img.get_data())

        n_voxels = data.shape[0]*data.shape[1]*data.shape[2]
        ts = data.reshape(n_voxels, data.shape[3])
        pca = PCA(50)
        pca.fit(ts)
        baseline = np.reshape(
            pca.components_[0, :], (len(pca.components_[0, :]), 1))
        new_ts = ts.T-np.dot(baseline, np.dot(np.linalg.pinv(baseline), ts.T))
        im2save = nib.Nifti1Image(
            new_ts.T.reshape(data.shape), affine=img.affine)
        nib.save(
            im2save, '{}_baseline_removed.nii.gz'.format(base))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume

        _, base, _ = split_filename(fname)

        outputs["detrended_file"] = os.path.abspath(
            '{}_baseline_removed.nii.gz'.format(base))

        return outputs


class SUVRCalculationInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='3D input file',
                  mandatory=True)
    base_mask = File(exists=True, desc='3D baseline mask',
                     mandatory=True)


class SUVRCalculationOutputSpec(TraitedSpec):

    SUVR_file = File(
        exists=True, desc='3D SUVR file')


class SUVRCalculation(BaseInterface):

    input_spec = SUVRCalculationInputSpec
    output_spec = SUVRCalculationOutputSpec

    def _run_interface(self, runtime):

        fname = self.inputs.volume
        maskname = self.inputs.base_mask
        _, base, _ = split_filename(fname)

        img = nib.load(fname)
        data = np.array(img.get_data())
        mask = nib.load(maskname)
        mask = np.array(mask.get_data())

        [x, y, z] = np.where(mask > 0)
        ii = np.arange(x.shape[0])
        mean_uptake = np.mean(data[x[ii], y[ii], z[ii]])
        new_data = data / mean_uptake
        im2save = nib.Nifti1Image(new_data, affine=img.affine)
        nib.save(im2save, '{}_SUVR.nii.gz'.format(base))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume

        _, base, _ = split_filename(fname)

        outputs["SUVR_file"] = os.path.abspath(
            '{}_SUVR.nii.gz'.format(base))

        return outputs


class PrepareUnlistingInputsInputSpec(BaseInterfaceInputSpec):

    time_offset = traits.Int(desc='Time between the PET start time and the '
                             'time when you want to initiate the sinogram '
                             'sorting (in seconds).')
    num_frames = traits.Int(desc='Number of frame you want to unlist.')
    temporal_len = traits.Float(desc='Temporal duration, in seconds, of each '
                                'frame. Minumum is 0.001.')
    list_mode = File(exists=True, desc='Listmode data')


class PrepareUnlistingInputsOutputSpec(TraitedSpec):

    out = traits.List(desc='List of all the outputs for PETListModeUnlisting '
                      'pipeline')


class PrepareUnlistingInputs(BaseInterface):

    input_spec = PrepareUnlistingInputsInputSpec
    output_spec = PrepareUnlistingInputsOutputSpec

    def _run_interface(self, runtime):

        time_offset = self.inputs.time_offset
        num_frames = self.inputs.num_frames
        temporal_len = self.inputs.temporal_len
        list_mode = self.inputs.list_mode

        start_times = (np.arange(time_offset, num_frames*temporal_len,
                                 temporal_len)).tolist()

        self.list_outputs = zip([list_mode]*len(start_times), start_times,
                                [temporal_len]*len(start_times))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["out"] = self.list_outputs

        return outputs


class PETListModeUnlistingInputSpec(BaseInterfaceInputSpec):

    list_inputs = traits.List(desc='List containing the time_offset and the '
                              'temporal frame length, as generated by '
                              'PrepareUnlistingInputs')


class PETListModeUnlistingOutputSpec(TraitedSpec):

    pet_sinogram = File(exists=True, desc='unlisted sinogram.')


class PETListModeUnlisting(BaseInterface):

    input_spec = PETListModeUnlistingInputSpec
    output_spec = PETListModeUnlistingOutputSpec

    def _run_interface(self, runtime):

        file_path = self.inputs.list_inputs[0]
        start = self.inputs.list_inputs[1]
        frame_len = self.inputs.list_inputs[2]
        end = start+frame_len
        print 'Unlisting Frame {}'.format(str(start/frame_len))
        cmd = (
            '{0} {1} 0 {2} 4 {3} {4} {5}'.format(
                list_mode_framing_path, file_path, str(frame_len), str(start),
                str(end), str(start/frame_len)))
        sp.check_output(cmd, shell=True)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        start = self.inputs.list_inputs[0]
        frame_len = self.inputs.list_inputs[1]
        fname = self.inputs.list_mode

        _, _, ext = split_filename(fname)

        outputs["pet_sinogram"] = (
            os.getcwd()+'/Frame{0}{1}'.format(str(start/frame_len).zfill(5),
                                              ext))

        return outputs


class SSRBInputSpec(BaseInterfaceInputSpec):

    unlisted_sinogram = File(exists=True, desc='unlisted sinogram, output of '
                             'PETListModeUnlisting.')


class SSRBOutputSpec(TraitedSpec):

    ssrb_sinogram = File(exists=True, desc='Sinogram compressed using SSRB '
                         'algorithm. This will be the input of the PCA method '
                         'for motion detection')


class SSRB(BaseInterface):

    input_spec = SSRBInputSpec
    output_spec = SSRBOutputSpec

    def _run_interface(self, runtime):

        unlisted_sinogram = self.inputs.unlisted_sinogram
        basename = unlisted_sinogram.split('.')[0]
        new_ext = '.s'
        new_name = basename+new_ext
        os.rename(unlisted_sinogram, new_name)
        self.gen_interfiles(new_name)
        num_segs_to_combine = 1
        view_mash = 36
        do_ssrb_norm = 0
        cmd = ('SSRB {0}_ssrb {0}.hs {1} {2} {3}'
               .format(basename, num_segs_to_combine, view_mash, do_ssrb_norm))
        sp.check_output(cmd, shell=True)
        os.remove(basename+'.hs')

        return runtime

    def gen_interfiles(self, sinogram):

        file_pre = sinogram.split('/')[-1].split('.')[0]
        file_suf = '.hs'
        print 'New file name: ' + file_pre + file_suf
        with open(file_pre + file_suf, 'w') as new_file:
            with open(interfile_path) as old_file:
                for line in old_file:
                    new_file.write(
                        line.replace('name of data file := biograph_mmr.s\n',
                                     'name of data file := ' +
                                     sinogram.split('/')[-1]+'\n'))

        new_file.close()
        old_file.close()

    def _list_outputs(self):
        outputs = self._outputs().get()
        unlisted_sinogram = self.inputs.unlisted_sinogram
        basename = unlisted_sinogram.split('.')[0]

        outputs["ssrb_sinogram"] = basename+'_ssrb.s'

        return outputs


class MergeUnlistingOutputsInputSpec(BaseInterfaceInputSpec):

    sinograms = traits.List(desc='List of ssrb sinogram to merge into'
                            'one folder.')


class MergeUnlistingOutputsOutputSpec(TraitedSpec):

    sinogram_folder = Directory(desc='Directory containing all the compressed '
                                'sinograms.')


class MergeUnlistingOutputs(BaseInterface):

    input_spec = MergeUnlistingOutputsInputSpec
    output_spec = MergeUnlistingOutputsOutputSpec

    def _run_interface(self, runtime):

        sinograms = self.inputs.sinograms
        if os.path.isdir('PET_sinograms_for_PCA') is False:
            os.mkdir('PET_sinograms_for_PCA')
        for s in sinograms:
            shutil.move(s, 'PET_sinograms_for_PCA')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["sinogram_folder"] = os.getcwd()+'/PET_sinograms_for_PCA'

        return outputs


class PreparePetDirInputSpec(BaseInterfaceInputSpec):

    pet_dir = Directory(exists=True, desc='Directory with the PET images to '
                        'use for motion correction.')
    image_orientation_check = traits.Bool(
        desc='This check is to assure that the PET images have the correct '
        'orientation (for example, using old version of e7tools the fineal PET'
        'images had wrong RL, AP and IS labels with respect to the actual '
        'image orientation). By default the pipeline checks the version of the'
        ' e7tools version used for the reconstrution in the PET header. If '
        'there is no information there and you are sure the orientation is '
        'correct then set this to True, otherwise recontruct the images with a'
        'new e7tools version. This software does not support old e7tools.',
        default=False)


class PreparePetDirOutputSpec(TraitedSpec):

    pet_dir_prepared = Directory(desc='Directory with PET images ready for the'
                                 ' motion correction.')


class PreparePetDir(BaseInterface):

    input_spec = PreparePetDirInputSpec
    output_spec = PreparePetDirOutputSpec

    def _run_interface(self, runtime):

        pet_dir = self.inputs.pet_dir
        image_orientation_check = self.inputs.image_orientation_check
        basename = 'frame'
        pet_images = sorted(
            glob.glob(pet_dir+'/{0}*.nii.gz'.format(basename)))

        if not pet_images:
            pet_images = sorted(
                glob.glob(pet_dir+'/{0}*.nii'.format(basename)))
        if pet_images:
            im = nib.load(pet_images[0])
            hd = im.header
            if 'New_e7tools' in hd['db_name']:
                image_orientation_check = True
                print ('New e7tool version detected.')
        if not pet_images:
            pet_dicoms = sorted(glob.glob(pet_dir + '/Frame*'))
            if pet_dicoms:
                vol0 = sorted(glob.glob(pet_dicoms[0]+'/*'))[0]
                hd = pydicom.read_file(vol0)
                if ('e7tools' in hd.SoftwareVersions or
                        'syngo MR B20P' in hd.SoftwareVersions or
                        'syngo MR E11' in hd.SoftwareVersions):
                    image_orientation_check = True
                    print ('New e7tool version detected.')
                for dcm in pet_dicoms:
                    frame_num = dcm.split('/')[-1][5:]
                    cmd = ('mrconvert -force {0} {1}/{2}{3}.nii.gz'
                           .format(dcm, pet_dir, basename,
                                   str(frame_num).zfill(3)))
                    sp.check_output(cmd, shell=True)
                    if frame_num == '0' and image_orientation_check:
                        im = nib.load('{0}/{1}{2}.nii.gz'.format(
                                pet_dir, basename, str(frame_num).zfill(3)))
                        hd = im.header
                        hd['db_name'] = 'New_e7tools'
                        nib.save(
                            im, '{0}/{1}{2}.nii.gz'.format(
                                pet_dir, basename, str(frame_num).zfill(3)))
                pet_images = sorted(glob.glob(
                    pet_dir+'/{0}*.nii.gz'.format(basename)))
            else:
                raise Exception("No PET images found in {0}!".format(pet_dir))
        if not image_orientation_check:
            raise Exception(
                "Could not find any e7tools version information and the PET "
                "header. If you are sure that the reconstructed PET images "
                "have correct orientation then specify image_orientation_check"
                "=True. Otherwise reconstruct your images with the new version"
                ". This software does not support the old e7tools version.")
        os.mkdir('pet_data')
        for f in pet_images:
            shutil.move(f, 'pet_data')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["pet_dir_prepared"] = os.getcwd()+'/pet_data'

        return outputs


class PETFovCroppingInputSpec(BaseInterfaceInputSpec):

    pet_image = File(exists=True, desc='PET images to crop.')
#     ref_pet = File(exists=True, desc='Reference image to use to save the '
#                    'cropped PET. Usually is the output of fslroi command '
#                    'with the same cropping parameters.')
    x_min = traits.Int()
    x_size = traits.Int()
    y_min = traits.Int()
    y_size = traits.Int()
    z_min = traits.Int()
    z_size = traits.Int()


class PETFovCroppingOutputSpec(TraitedSpec):

    pet_cropped = File(exists=True, desc='Cropped PET')


class PETFovCropping(BaseInterface):

    input_spec = PETFovCroppingInputSpec
    output_spec = PETFovCroppingOutputSpec

    def _run_interface(self, runtime):

        pet_image = self.inputs.pet_image
#         ref = self.inputs.ref_pet
        x_min = self.inputs.x_min
        x_size = self.inputs.x_size
        y_min = self.inputs.y_min
        y_size = self.inputs.y_size
        z_min = self.inputs.z_min
        z_size = self.inputs.z_size
        _, basename, ext = split_filename(pet_image)
        outname = basename+'_crop'+ext
        pet = nib.load(pet_image)
        new_affine = np.copy(pet.affine)
        new_affine[:3, -1] = (pet.affine[:3, -1]-np.multiply(
            pet.header.get_zooms(), (x_min, y_min, z_min)) *
            np.sign(pet.affine[:3, -1]))
        pet = pet.get_data()
        pet_cropped = pet[x_min:x_min+x_size, y_min:y_min+y_size,
                          z_min:z_min+z_size]
#         cmd = 'fslroi {} ref_roi 100 130 100 130 20 100'.format(im)
#         sp.check_output(cmd, shell=True)
#         ref = nib.load(ref)
        im2save = nib.Nifti1Image(pet_cropped, affine=new_affine)
        im2save.set_qform(new_affine, code='scanner')
        im2save.set_sform(new_affine, code='scanner')
        nib.save(im2save, outname)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        pet_image = self.inputs.pet_image
        _, basename, ext = split_filename(pet_image)
        outname = basename+'_crop'+ext

        outputs["pet_cropped"] = os.getcwd()+'/'+outname

        return outputs


class StaticMotionCorrectionInputSpec(BaseInterfaceInputSpec):

    pet_cropped = Directory(desc='Directory with the fov cropped PET images.')
    md_dir = Directory(desc='Directory with the outputs from the MR-based '
                       'motion detection pipeline.')
    structural_image = File(desc='If provided, the final PET mc image will be '
                            'aligned to this image.', default=None)
    ute2structural_regmat = File(default=None)
    ute2structural_qform = File(default=None)


class StaticMotionCorrectionOutputSpec(TraitedSpec):

    pet_mc_results = Directory(
        entries=True, desc='Motin corrected static PET results.')
    pet_mc_results_to_crop = Directory(
        entries=True, desc='Motin corrected static PET results to crop (those '
        'in PET space).')


class StaticMotionCorrection(BaseInterface):

    input_spec = StaticMotionCorrectionInputSpec
    output_spec = StaticMotionCorrectionOutputSpec

    def _run_interface(self, runtime):

        md_dir = self.inputs.md_dir
        structural_image = self.inputs.structural_image
        ute2structural_regmat = self.inputs.ute2structural_regmat
        ute2structural_qform = self.inputs.ute2structural_qform
        pet_data = sorted(glob.glob(self.inputs.pet_cropped+'/*.nii.gz'))
        list_frame_mat = sorted(glob.glob(
            md_dir+'/motion_detection_frame2reference_mats/Frame*_inv.mat'))
        corr_factors = np.loadtxt(
            md_dir+'/motion_detection_correction_factors.txt')
        if not pet_data:
            raise Exception('No images found in {}!'
                            .format(self.inputs.pet_cropped))
        elif pet_data and len(pet_data) != len(list_frame_mat):
            raise Exception("The number of the PET images found in {0} is "
                            "different from that of the motion matrices found "
                            "in {1}. Please check."
                            .format(self.inputs.pet_cropped, md_dir +
                                    '/motion_detection_frame2reference_mats'))
        try:
            ute = sorted(glob.glob(md_dir+'/*ute*.nii*'))[0]
        except IndexError:
            raise Exception('No UTE image found in {}. UTE image is needed to '
                            'run motion correction. Please add one UTE image '
                            '(with name matching *ute*.nii*) in the motion '
                            'detection directory or run the MR-based motion'
                            'detection again providing one UTE image.'
                            .format(md_dir))

        ute_qform = self.extract_qform(ute)
        pet_qform = self.extract_qform(pet_data[0])
        fixed_MRPET_transformation = np.dot(ute_qform,
                                            np.linalg.inv(pet_qform))
        fixed_MRPET_transformation_inv = np.linalg.inv(
            fixed_MRPET_transformation)
        curr_dir = os.getcwd()
        os.mkdir('mc_results')
        os.chdir('mc_results')
        np.savetxt('MRPET_transformation.mat', fixed_MRPET_transformation)
        for i in range(len(pet_data)):
            basename = pet_data[i].split('/')[-1].split('.')[0]
            print 'Aligning {}...'.format(basename)
            motion_mat = np.loadtxt(list_frame_mat[i])
            transformation_mat = (
                np.dot(motion_mat, fixed_MRPET_transformation))
            transformation_mat_petspace = (
                np.dot(fixed_MRPET_transformation_inv, transformation_mat))
            if structural_image:
                transformation_mat = (np.dot(ute2structural_regmat,
                                             transformation_mat))
                ref = structural_image
                out_basename = 'al2Struct'
                self.applyxfm(pet_data[i], ref, ute2structural_qform,
                              '{0}_{1}_no_mc'.format(basename, out_basename))
            else:
                ref = ute
                out_basename = 'al2UTE'
                self.applyxfm(pet_data[i], ref, 'MRPET_transformation.mat',
                              '{0}_{1}_no_mc'.format(basename, out_basename))
            outname = '{0}_{1}'.format(basename, out_basename)
            np.savetxt('transformation_{0}.mat'.format(str(i)),
                       transformation_mat)
            np.savetxt('transformation_{0}_ps.mat'.format(str(i)),
                       transformation_mat_petspace)
            self.applyxfm(
                pet_data[i], ref, 'transformation_{}.mat'.format(str(i)),
                outname+'_mc')
            self.applyxfm(
                pet_data[i], pet_data[i],
                'transformation_{}_ps.mat'.format(str(i)), outname+'_mc_ps')
            corr_types = ['_mc', '_mc_ps', '_no_mc']
            for tps in corr_types:
                self.apply_temporal_correction(outname, corr_factors[i], tps)
        print 'Generating static PET images...'
        corr_types = ['mc_corr', 'mc_ps_corr', 'no_mc_corr']
        for tps in corr_types:
            self.frames_sum(tps, out_basename)
        os.mkdir('images2crop')
        shutil.move('frame_mc_ps_corr.nii.gz', 'images2crop')
        os.chdir(curr_dir)

        return runtime

    def frames_sum(self, im_type, outname):

        cmd = 'fslmaths '
        frames = glob.glob('frame*{0}_{1}.nii.gz'.format(outname, im_type))
        for i, frame in enumerate(frames):
            if i != len(frames)-1:
                cmd = cmd + '{0} -add '.format(frame)
            else:
                cmd = (cmd+'{0} frame_{1}'.format(frame, im_type))
        sp.check_output(cmd, shell=True)

    def apply_temporal_correction(self, image, corr_factor, tps):
        if tps:
            outname = image+tps+'_corr'
        else:
            outname = image+'_no_mc_ps_corr'
            image = image.split('_al2')[0]
        cmd = ('fslmaths {0}.nii.gz -mul {1} {2}'
               .format(image+tps, corr_factor, outname))
        sp.check_output(cmd, shell=True)

    def applyxfm(self, in_file, ref, mat, outname):

        applyxfm = fsl.FLIRT()
        applyxfm.inputs.in_file = in_file
        applyxfm.inputs.reference = ref
        applyxfm.inputs.apply_xfm = True
        applyxfm.inputs.in_matrix_file = mat
        applyxfm.inputs.out_file = outname + '.nii.gz'
        applyxfm.run()

    def extract_qform(self, image):

        cmd = 'fslhd {}'.format(image)
        image_info = sp.check_output(cmd, shell=True).strip().split('\n')
        qform = np.eye(4)
        for i, line in enumerate(image_info):
            if 'qto_xyz:1' in line:
                qform[0, -1] = abs(float(
                    image_info[i].split(':1')[-1].strip().split()[-1]))
                qform[1, -1] = abs(float(
                    image_info[i+1].split(':2')[-1].strip().split()[-1]))
                qform[2, -1] = abs(float(
                    image_info[i+2].split(':3')[-1].strip().split()[-1]))
                break

        return qform

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["pet_mc_results"] = os.getcwd()+'/mc_results'
        outputs["pet_mc_results_to_crop"] = (
            os.getcwd()+'/mc_results/images2crop')
        return outputs


# class PETImageAlignment(BaseInterfaceInputSpec):
