
Installation
============

Python Package
--------------

Banana is a pure Python 3 package, which can be installed from the `Python
Package Index <http://pypi.org>`_ using *Pip3*::

    $ pip3 install banana

Neuroimaging Toolkits
---------------------

Depending on the analyses you would like to run you will need to install
various Neuroimaging toolkits e.g:

* `MRtrix3 <https://github.com/MRtrix3/mrtrix3>`_
* `Dcm2niix <https://github.com/rordenlab/dcm2niix>`_
* `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_
* `Matlab <https://au.mathworks.com/products/matlab.html>`_
* `SPM <http://www.fil.ion.ucl.ac.uk/spm/software/spm12/>`_
* `ANTs <https://sourceforge.net/projects/advants/>`_
* `Freesurfer <https://surfer.nmr.mgh.harvard.edu>`_
* `AFNI <https://afni.nimh.nih.gov>`_

These toolkits can either be installed and accessible on your `PATH
<https://en.wikipedia.org/wiki/PATH_(variable)>`_ or installed in `environment
modules <https://en.wikipedia.org/wiki/Environment_Modules_(software)>`_ (as is
typical on many high-performance computer clusters). One advantage of
installing toolkits in separate environment modules is that you use different
versions of the same tool in different nodes of the pipeline.

For a given Study class and requested derivative(s) Banana will determine the
toolkits required and check they are available on If toolkits are 
