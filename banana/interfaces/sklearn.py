
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec)
import nibabel as nib
import numpy as np
from sklearn.decomposition import FastICA as fICA
from nipype.utils.filemanip import split_filename
import os


class FastICAInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='4D file to be decomposed using ICA',
                  mandatory=True)
    n_components = traits.Int(desc='Number of ICA components to extract',
                              mandatory=True)
    ica_type = traits.Str(desc='Type of ICA to run. Possible types are '
                          'spatial (default) and temporal.', default='spatial')


class FastICAOutputSpec(TraitedSpec):

    ica_decomposition = File(
        exists=True, desc='Nifti file containing ICA decomposition')
    ica_timeseries = File(
        exists=True, desc='Nifti file containing ICA timecourse plots')
    mixing_mat = File(
        exists=True, desc='Text file containing ICA mixing matrix')


class FastICA(BaseInterface):
    input_spec = FastICAInputSpec
    output_spec = FastICAOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.volume
        img = nib.load(fname)
        comp = self.inputs.n_components
        data = np.array(img.get_data())

        n_voxels = data.shape[0]*data.shape[1]*data.shape[2]
        ts = data.reshape(n_voxels, data.shape[3])
        _, base, _ = split_filename(fname)

        if self.inputs.ica_type == 'spatial':
            ica_input = ts.T
            outname = 'sICA'
        else:
            ica_input = ts
            outname = 'tICA'

        # Run ICA
        ica = fICA(n_components=comp)
        ica.fit(ica_input)
        S_ = ica.fit_transform(ica_input)
        if self.inputs.ica_type == 'spatial':
            sm = ica.components_.T[:]
            tc = S_[:]
        else:
            sm = S_[:]
            tc = ica.components_.T[:]

        ica_zscore = np.zeros((data.shape[0], data.shape[1],
                               data.shape[2], self.inputs.n_components))
        ica_tc = np.zeros((data.shape[3], self.inputs.n_components))
        for i in range(self.inputs.n_components):
            dt = sm[:, i]-np.mean(sm[:, i])
            num = np.mean(dt**3)
            denom = (np.mean(dt**2))**1.5
            s = num / denom
            print(s)
            if np.sign(s) == -1:
                print('Flipping sign of component {}'.format(str(i)))
                sm[:, i] = -1*sm[:, i]
                tc[:, i] = -1*tc[:, i]

            pc = sm[:, i].reshape(
                data.shape[0], data.shape[1], data.shape[2])

            vstd = np.linalg.norm(sm[:, i])/np.sqrt(n_voxels-1)
            if vstd != 0:
                pc_zscore = pc/vstd
            else:
                print ('Not converting to z-scores as division by zero'
                       ' warning may occur.')
                pc_zscore = pc
            ica_zscore[:, :, :, i] = pc_zscore
            ica_tc[:, i] = tc[:, i]

        im2save = nib.Nifti1Image(ica_zscore, affine=img.affine)
        tc2save = nib.Nifti1Image(ica_tc, affine=np.eye(4))
        nib.save(
            im2save, '{0}_{1}_results_pc{2}_zscore.nii.gz'
            .format(base, outname, str(self.inputs.n_components)))
        nib.save(
            tc2save, '{0}_{1}_timecourse_pc{2}.nii.gz'
            .format(base, outname, str(self.inputs.n_components)))
        np.savetxt(
            '{0}_{1}_mixing_matrix_pc{2}.txt'.format(
                base, outname, str(self.inputs.n_components)), S_)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume
        if self.inputs.ica_type == 'spatial':
            outname = 'sICA'
        else:
            outname = 'tICA'
        _, base, _ = split_filename(fname)
        outputs["ica_decomposition"] = os.path.abspath(
            base+'_{0}_results_pc{1}_zscore.nii.gz'.format(
                outname, str(self.inputs.n_components)))
        outputs["ica_timeseries"] = os.path.abspath(
            base+'_{0}_timecourse_pc{1}.nii.gz'.format(
                outname, str(self.inputs.n_components)))
        outputs["mixing_mat"] = os.path.abspath(
            base+'_{0}_mixing_matrix_pc{1}.txt'.format(
                outname, str(self.inputs.n_components)))

        return outputs
