NiAnalysis
==========

A collection of Arcana_ "Study" classes that implement various neuroimaging
analysis workflows, which can be 

Dependencies
-----------------

Core Python
~~~~~~~~~~~

* arcana
* pydicom

Both of these are available on PyPI and can be installed with::

    $ pip install -e <path-to-nianalysis-repo>


Pipeline
~~~~~~~~

Depending on which pipelines you may need to install some or all of the following.
Note that MRtrix3 and Dcm2niix are used for implicit format conversions so are
typically required.

* MRtrix3 (https://github.com/MRtrix3/mrtrix3)
* Dcm2niix (https://github.com/rordenlab/dcm2niix)
* FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
* Matlab (https://au.mathworks.com/products/matlab.html)
* SPM (http://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
* ANTs (https://sourceforge.net/projects/advants/)
* Freesurfer (https://surfer.nmr.mgh.harvard.edu)
* AFNI (https://afni.nimh.nih.gov)

.. _Arcana: http://github.com/monashbiomedicalimaging/arcana
