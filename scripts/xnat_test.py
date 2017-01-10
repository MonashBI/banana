import os.path
import xnat
from nianalysis.testing import test_data_dir

TEST_IMAGE = os.path.abspath(os.path.join(test_data_dir, 'test_image.nii.gz'))

mbi_xnat = xnat.connect('https://mbi-xnat.erc.monash.edu.au', user='unittest',
                        password='Test123!')

exp = mbi_xnat.projects['TEST001'].subjects['TEST001_ARCHIVEXNAT'].experiments[
    'TEST001_ARCHIVEXNAT_XNATARCHIVE']

# try:
#     exp.scans['source'].delete()
# except:
#     pass
# dataset = mbi_xnat.classes.MrScanData(parent=exp, type='source')
dataset = exp.scans['source']
resource = dataset.resources['NIFTI_GZ']

# resource = dataset.create_resource('NIFTI_GZ')
# resource.upload(TEST_IMAGE, os.path.join('NIFTI_GZ', 'source.nii.gz'))

print resource.files