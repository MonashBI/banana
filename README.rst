Banana
======

.. image:: https://img.shields.io/pypi/pyversions/banana.svg
  :target: https://pypi.python.org/pypi/banana/
  :alt: Supported Python versions
.. image:: https://img.shields.io/pypi/v/banana.svg
  :target: https://pypi.python.org/pypi/banana/
  :alt: Latest Version    
.. image:: https://readthedocs.org/projects/banana/badge/?version=latest
  :target: http://banana.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status


Biomedical imaging ANAlysis iN Arcana (Banana): a collection of biomedical analysis
workflows implemented in the Arcana_ framework, which can be used to analyse
study datasets stored in XNAT, BIDS or plain-directory repositories.

Dependencies
-----------------

Core
~~~~

* arcana
* nibabel
* pydicom

All of these are available on PyPI and can be installed with::

    $ pip3 install banana


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
