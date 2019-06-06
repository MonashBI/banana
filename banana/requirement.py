from future.utils import PY3
import os
import os.path as op
import platform
import re
import xml.etree.ElementTree
from arcana.environment.requirement import (
    Version, CliRequirement, MatlabPackageRequirement,
    PythonPackageRequirement,
    matlab_req)  # @UnusedImport
from arcana.utils import run_matlab_cmd
from arcana.exceptions import (
    ArcanaRequirementNotFoundError, ArcanaVersionNotDetectableError)

# Command line requirements


class FSLRequirement(CliRequirement):

    def detect_version_str(self):
        """
        As FSL doesn't have a simple way of printing the version, we need to
        read the version from the 'fslversion' text file in the FSLDIR.
        """
        try:
            fsl_dir = os.environ['FSLDIR']
        except KeyError:
            raise ArcanaRequirementNotFoundError(
                "Could not find FSL, 'FSLDIR' environment variable is not set")
        with open(op.join(fsl_dir, 'etc', 'fslversion'), 'r') as f:
            contents = f.read()
        return contents.strip()


class C3dRequirement(CliRequirement):

    def detect_version_str(self):
        """
        C3D doesn't print out the version so we need to interrogate the
        install directory to extract it
        """
        c3d_bin_path = op.dirname(self.locate_command())
        if platform.system() == 'Linux':
            libname = os.listdir(op.join(c3d_bin_path, '..', 'lib'))[0]
            version_str = libname.split('-')[-1]
        elif platform.system() == 'Darwin':
            info_list_path = op.join(c3d_bin_path, '..', 'Info.plist')
            info_etree = xml.etree.ElementTree.parse(info_list_path)
            elem_bodies = [e.text for e in info_etree.iter()]
            version_str = elem_bodies[
                elem_bodies.index('CFBundleShortVersionString') + 1]
        else:
            raise ArcanaVersionNotDetectableError(
                "Can't detect c3d version on Windows")
        return version_str


class FreesurferRequirement(CliRequirement):

    def detect_version_str(self):
        """
        The version that recon-all spits out doesn't match that of Freesurfer
        itself so we take it from the build-stamp.txt file instead
        """
        try:
            fs_dir = os.environ['FREESURFER_HOME']
        except KeyError:
            raise ArcanaRequirementNotFoundError(
                "Could not find Freesurfer installation as 'FREESURFER_HOME' "
                "environment variable is not set")
        with open(op.join(fs_dir, 'build-stamp.txt'), 'r') as f:
            contents = f.read()
        return re.match(r'freesurfer-.*-v(.*)', contents).group(1)


class MrtrixRequirement(CliRequirement):

    def detect_version_str(self):
        version_str = super().detect_version_str()
        return re.match(r'== mrinfo (.*) ==', version_str).group(1)


class AfniRequirement(CliRequirement):

    def detect_version_str(self):
        version_str = super().detect_version_str()
        return re.match(r'.*AFNI_([\d\.]+)', version_str).group(1)


class FixVersion(Version):

    regex = re.compile(r'(\d+)\.(\d)?(\d)?(\d)?')

    def __str__(self):
        return '{}.{}'.format(self._seq[0], ''.join(self._seq[1:]))


class StirRequirement(CliRequirement):

    def detect_version_str(self):
        raise ArcanaVersionNotDetectableError(
            "Can't automatically detect version of STIR as it isn't saved in "
            "the build process")


mrtrix_req = MrtrixRequirement('mrtrix', test_cmd='mrinfo')
ants_req = CliRequirement('ants', test_cmd='antsRegistration')
dcm2niix_req = CliRequirement('dcm2niix', test_cmd='dcm2niix')
freesurfer_req = FreesurferRequirement('freesurfer', test_cmd='recon-all')
fix_req = CliRequirement('fix', test_cmd='fix', version_cls=FixVersion)
afni_req = AfniRequirement('afni', test_cmd='afni')
fsl_req = FSLRequirement('fsl', test_cmd='fslinfo')
c3d_req = C3dRequirement('c3d', test_cmd='c3d')
stir_req = StirRequirement('stir', test_cmd='SSRB')

# Matlab package requirements


class SpmRequirement(MatlabPackageRequirement):

    def parse_help_text(self, help_text):
        """
        Detect which SPM version we are using from the copyright year
        """
        match = re.search(
            r'Copyright \(C\) [\d\-\, ]*(?<!\d)(\d+) Wellcome Trust Centre',
            help_text)
        if match is None:
            raise ArcanaVersionNotDetectableError(
                "Could not parse year of copyright from spm_authors in order "
                "to determine the version of {}".format(self))
        copyright_year = match.group(1)
        if copyright_year == '2010':
            version = 8
        elif copyright_year == '2012':
            version = 12
        else:
            raise ArcanaVersionNotDetectableError(
                "Do not know the version of SPM corresponding to the year of "
                "copyright of {}".format(copyright_year))
        return version


class StiRequirement(MatlabPackageRequirement):

    def detect_version_str(self):
        """
        Detect which version by scanning the README for the latest release
        """
        cmd_path = run_matlab_cmd("which('{}')".format(self.test_func))
        pkg_root = op.join(op.dirname(cmd_path), '..')
        readmes = [f for f in os.listdir(pkg_root) if 'readme' in f.lower()]
        if not readmes:
            raise ArcanaVersionNotDetectableError(
                "Did not find a README in STI package root ({})"
                .format(pkg_root))
        elif len(readmes) > 1:
            raise ArcanaVersionNotDetectableError(
                "Found multiple READMEs in STI package root ({})"
                .format(pkg_root))
        readme_path = op.join(pkg_root, readmes[0])
        with open(readme_path, 'rb') as f:
            contents = f.read()
        # Cut out citations text as there can be non-decodable characters in
        # there
        contents = contents.split(b'TERMS OF USE')[0]
        if PY3:
            contents = contents.decode('utf-8')
        # Get dummy version so we can use its 'regex' property
        dummy_version_obj = self.version_cls(self, 1)
        versions = dummy_version_obj.regex.findall(contents)
        latest_version = sorted(versions)[-1]
        return latest_version


spm_req = SpmRequirement('spm', test_func='spm_authors')
sti_req = StiRequirement('sti', test_func='V_SHARP')
# noddi_req = MatlabPackageRequirement('noddi')


# Python package requirements

sklearn_req = PythonPackageRequirement('sklearn')
pydicom_req = PythonPackageRequirement('pydicom')
scipy_req = PythonPackageRequirement('scipy')
