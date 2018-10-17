import os.path as op
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File

from banana.interfaces import MATLAB_RESOURCES


class BaseVeinInputSpec(MatlabInputSpec):

    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class BaseVeinOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="Processed file")
    raw_output = traits.Str("Raw output of the matlab command")


class BaseVein(MatlabCommand):
    """
    Base class for MATLAB mask interfaces
    """
    output_spec = BaseVeinOutputSpec

    def run(self, **inputs):
        self.work_dir = inputs['cwd']
        # Set the script input of the matlab spec
        self.inputs.script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES) +
            self.script(**inputs) +
            "exit;")
        results = super().run(**inputs)
        stdout = results.runtime.stdout
        # Attach stdout to outputs to access matlab results
        results.outputs.raw_output = stdout
        return results

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to perform masking
        """
        raise NotImplementedError

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

    @property
    def out_file(self):
        return op.realpath(op.abspath(op.join(self.work_dir, 'out_file.nii')))


class ShMRFInputSpec(BaseVeinInputSpec):
    in_file = File(exists=True, mandatory=True)
    mask = File(exists=True, mandatory=True)
    omega1 = traits.Float(0.01, usedefault=True)
    omega2 = traits.Float(0.20, usedefault=True)


class ShMRF(BaseVein):
    input_spec = ShMRFInputSpec

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = """
            % Load input files
            cv_image = load_untouch_nii('{in_file}');
            mask = load_untouch_nii('{mask}');
            out = cv_image;

            mask = mask.img>0;
            cv_image = cv_image.img;
            cv_image(isnan(cv_image)) = 0;

            %%
            params = ShMRF_DefaultParams();
            params.omega1 = {omega1};
            params.omega2 = {omega2}; %(0.12 in first round)
            params.display = false;
            params.preprocess = false;

            out.img = ShMRF_Segment(cv_image, mask, params);

            save_untouch_nii(out, '{out_file}');
        """.format(
            in_file=self.inputs.in_file,
            mask=self.inputs.mask,
            omega1=self.inputs.omega1,
            omega2=self.inputs.omega2,
            out_file=self.out_file)
        return script


class CompositeVeinImageInputSpec(BaseVeinInputSpec):
    qsm = File(exists=True, mandatory=True)
    swi = File(exists=True, mandatory=True)
    mask = File(exists=True, mandatory=True)
    vein_atlas = File(exists=True, mandatory=True)
    q_prior = File(exists=True, mandatory=True)
    s_prior = File(exists=True, mandatory=True)
    a_prior = File(exists=True, mandatory=True)


class CompositeVeinImage(BaseVein):

    input_spec = CompositeVeinImageInputSpec

    def script(self, **inputs):  # @UnusedVariable
        script = """
        mask = load_untouch_nii('{mask}');
        mask = single(mask.img)>0;

        swi = load_untouch_nii('{swi}');
        qsm = load_untouch_nii('{qsm}');
        fre = load_untouch_nii('{vein_atlas}');

        hdrInfo = qsm.hdr;
        hdrInfo.dime.datatype = 64;
        hdrInfo.dime.bitpix = 64;

        swi = single(swi.img);
        qsm = single(qsm.img);
        fre = single(fre.img);

        [ swi, qsm ] = GMM( mask, swi, qsm);
        vein_atlas = min(0.99,max(0.01,fre));

        s_prior = load_untouch_nii('{s_prior}');
        q_prior = load_untouch_nii('{q_prior}');
        a_prior = load_untouch_nii('{a_prior}');

        s_prior = single(s_prior.img);
        q_prior = single(q_prior.img);
        a_prior = single(a_prior.img);

        cvVol = swi.*s_prior + qsm.*q_prior + vein_atlas.*a_prior;
        cvVol = cvVol./(s_prior+q_prior+a_prior);

        cvNii = make_nii(cvVol.*mask);
        cvNii.hdr = hdrInfo;
        save_nii(cvNii, '{out_file}');
        """.format(
            qsm=self.inputs.qsm,
            swi=self.inputs.swi,
            vein_atlas=self.inputs.vein_atlas,
            mask=self.inputs.mask,
            q_prior=self.inputs.q_prior,
            s_prior=self.inputs.s_prior,
            a_prior=self.inputs.a_prior,
            out_file=self.out_file)
        return script
