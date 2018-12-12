import numpy as np
from itertools import chain, repeat
from copy import copy
import os.path as op
import matplotlib.image
import tempfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import subprocess as sp
from banana.exceptions import BananaUsageError
from banana.file_format import nifti_gz_format, nifti_format, dicom_format


class ImageDisplayMixin():

    def display_slice_panel(self, filesets, img_size=5,
                            row_kwargs=None, offset=None, **kwargs):
        """
        Displays an image in a Nx3 panel axial, coronal and sagittal
        slices for the filesets correspdong to each of the data names
        provided.

        Parameters
        ----------
        data_names : List[str]
            List of image names to plot as rows of a panel
        size : Tuple(2)[int]
            Size of the figure to plot
        row_kargs : List[Dict[str, *]]
            A list of row-specific kwargs to passed on to
            _display_mid_slices
        offset : Tuple(3)[int]
            An array of integers with which to offset the slices displayed
        """
        n_rows = len(filesets)
        if row_kwargs is None:
            row_kwargs = repeat({})
        elif not n_rows == len(row_kwargs):
            raise BananaUsageError(
                "Length of row_kwargs ({}) needs to "
                "match length of filesets ({})"
                .format(len(row_kwargs), n_rows))
        # Set up figure
        gs = GridSpec(n_rows, 3)
        gs.update(wspace=0.0, hspace=0.0)
        fig = plt.figure(figsize=(3 * img_size,
                                  n_rows * img_size))
        # Loop through derivatives and generate image
        for i, (fileset, rkwargs) in enumerate(zip(filesets,
                                                   row_kwargs)):
            array = fileset.get_array()
            header = fileset.get_header()
            if fileset.format in (nifti_format, nifti_gz_format):
                vox = header['pixdim'][1:4]
            elif fileset.format == dicom_format:
                vox = [float(v) for v in header.PixelSpacing]
                vox.append(float(header.SliceThickness))
            else:
                raise BananaUsageError(
                    "'{}' format images are not supported for display slice "
                    .format(fileset.format))
            rkwargs = copy(rkwargs)
            rkwargs.update(kwargs)
            try:
                self._display_mid_slices(array, vox, fig, gs, i,
                                         offset=offset, **rkwargs)
            except BananaUsageError as e:
                raise BananaUsageError(
                    str(e) + " displaying {}".format(fileset.path))
        # Remove space around figure
        plt.tight_layout(0.0)
        # Either show image or save it to file

    def display_tcks_with_mrview(self, tcks, backgrounds, padding=1,
                                  img_size=5, offset=None):
        """
        Displays dMRI tractography streamlines using MRtrix's mrview to
        display and screenshot to file the streamlines. Then reload,
        crop and combine the different slice orientations into
        a single panel.

        Parameters
        ----------
        save_path : str | None
            The path of the file to save the image at. If None the
            images are displayed instead
        size : Tuple(2)[int]
            The size of the combined figure to plot
        offset : Tuple(3)[int]
            An array of integers with with to offset the slices displayed
        """
        n_rows = len(tcks)
        # Create figure in which to aggregate the plots
        gs = GridSpec(n_rows, 3)
        gs.update(wspace=0.0, hspace=0.0)
        fig = plt.figure(figsize=(3 * img_size,
                                  n_rows * img_size))
        # Create temp dir to store screen captures
        tmpdir = tempfile.mkdtemp()
        for i, (tck, bg) in enumerate(zip(tcks, backgrounds)):
            # Call mrview to display the tracks, capture them to
            # file and then reload the image into a matplotlib
            # grid
            options = ['-tractography.load', tck.path, '-noannotations']
            if offset is not None:
                array = bg.get_array()
                centre = self.image_centre(array, offset)
                options.extend(['-voxel', '{},{},{}'.format(*centre)])
            # Set options to remove cursor, capture hte image and exit
            options.extend(['-lock', 'yes', '-capture.grab', '-exit'])
            imgs = []
            for i in range(3):
                sp.call(
                    '{} {} -plane {} {}'.format(
                        self.mrview_path, bg.path, 2 - i,
                        ' '.join(options)), cwd=tmpdir,
                    shell=(self.mrview_path == 'mrview'))
                img = matplotlib.image.imread(
                    op.join(tmpdir, 'screenshot0000.png'))
                imgs.append(self.crop(img, border=padding))
            padded_size = max(chain(*(a.shape for a in imgs)))
            for i, img in enumerate(imgs):
                img = self.pad_to_size(img, (padded_size,
                                              padded_size))
                axis = fig.add_subplot(gs[i])
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)
                plt.imshow(img)
        plt.tight_layout(0.0)

    def show(self, path=None, subj_id=None, visit_id=None):
        """
        Save current figure to file or show it
        """
        if path is None:
            plt.show()
        else:
            base, ext = op.splitext(path)
            if len(list(self.subject_ids)) > 1:
                base += '-sub{}'.format(subj_id)
            if len(list(self.visit_ids)) > 1:
                base += '-vis{}'.format(visit_id)
            plt.savefig(base + ext)

    def _display_mid_slices(self, array, vox_sizes, fig, grid_spec,
                            row_index, padding=1, vmax=None, vmin=None,
                            vmax_percentile=98, vmin_percentile=2,
                            offset=None):
        # Guard agains NaN
        array[np.isnan(array)] = 0.0
        # Crop black-space around array
        array = self.crop(array, padding)
        # Pad out image array into cube
        padded_size = np.max(array.shape)
        # Get centre of the image in order to take image slices there
        centre = self.image_centre(array, offset)
        # Get dynamic range of array
        if vmax is None:
            assert vmax_percentile is not None
            vmax = np.percentile(array, vmax_percentile)
        if vmin is None:
            assert vmin_percentile is not None
            vmin = np.percentile(array, vmin_percentile)

        # Function to plot a slice
        def display_slice(slce, index, aspect):
            axis = fig.add_subplot(grid_spec[index])
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            pad_vert = int(np.round(padded_size / aspect))
            padded_slce = self.pad_to_size(slce, (pad_vert, padded_size))
            plt.imshow(padded_slce,
                       interpolation='bilinear',
                       cmap='gray', aspect=aspect,
                       vmin=vmin, vmax=vmax)
        # Display slices
        display_slice(np.squeeze(array[-1:0:-1, -1:0:-1, centre[2]]).T,
                      row_index * 3, vox_sizes[1] / vox_sizes[0])
        display_slice(np.squeeze(array[-1:0:-1, centre[1], -1:0:-1]).T,
                      row_index * 3 + 1, vox_sizes[2] / vox_sizes[0])
        display_slice(np.squeeze(array[centre[0], -1:0:-1, -1:0:-1]).T,
                      row_index * 3 + 2, vox_sizes[2] / vox_sizes[1])

    @classmethod
    def crop(cls, array, border=0):
        """
        Crops an image array to the nonzero voxels (+ equal border in
        each dimension)
        """
        nonzeros = np.argwhere(array)
        # Build index slices for each dimension of the array
        slices = []
        for i, (a, b) in enumerate(zip(nonzeros.min(axis=0),
                                       nonzeros.max(axis=0))):
            # Append the index slice for the
            slices.append(slice(max(a - border, 0),
                                min(b + border, array.shape[i])))
        return array[tuple(slices)]

    @classmethod
    def image_centre(cls, array, offset=None):
        """
        Returns the centre point of the non-zero voxels in the image
        """
        nonzeros = np.argwhere(array)
        min_ind = nonzeros.min(axis=0)
        max_ind = nonzeros.max(axis=0)
        centre = ((max_ind - min_ind) // 2) + min_ind
        if offset is not None:
            centre += np.array(offset, dtype=int)
            if np.any(centre < 0) or np.any(centre > array.shape):
                raise BananaUsageError(
                    "Specified offset ({}) is larger than the "
                    "dimension of the image / 2 ({})"
                    .format(offset, array.shape // 2))
        return centre

    @classmethod
    def pad_to_size(cls, array, size):
        pad_before = [
            (size[i] - array.shape[i]) // 2
            for i in range(2)]
        pad_after = [
            size[i] - array.shape[i] - pad_before[i]
            for i in range(2)]
        padding = list(zip(pad_before, pad_after))
        padding.extend((0, 0) for _ in range(array.ndim - 2))
        return np.pad(array, padding, 'constant')

    @property
    def mrview_path(self):
        """
        The path to the 'mrview' executable. Returns simply 'mrview'
        if it hasn't been explicitly, which assumes it is on the path
        """
        try:
            return self._mrview_path
        except AttributeError:
            return 'mrview'

    @mrview_path.setter
    def mrview_path(self, path):
        self._mrview_path = path
