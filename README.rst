Banana
======

Biomedical imAgiNg ANAlysis (Banana): a collection of biomedical imaging analysis workflows
implemented in the Arcana_ framework, which can be used to analyse complete study
datasets in XNAT, BIDS or plain-directory repositories.

Dependencies
-----------------

Core
~~~~

* arcana
* nibabel
* pydicom

Both of these are available on PyPI and can be installed with::

    $ pip install banana


Pipelines
~~~~~~~~~

Depending on which pipelines you need to run, you may need to install some or
all of the following tools. Note that MRtrix3 and Dcm2niix are used for implicit
format conversions so are typically required.

* MRtrix3 (https://github.com/MRtrix3/mrtrix3)
* Dcm2niix (https://github.com/rordenlab/dcm2niix)
* FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
* Matlab (https://au.mathworks.com/products/matlab.html)
* SPM (http://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
* ANTs (https://sourceforge.net/projects/advants/)
* Freesurfer (https://surfer.nmr.mgh.harvard.edu)
* AFNI (https://afni.nimh.nih.gov)

.. _Arcana: http://arcana.readthedocs.io
