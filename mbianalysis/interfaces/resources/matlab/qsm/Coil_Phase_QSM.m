function Coil_Phase_QSM( inDir, rawDir, maskDir, outDir )
%COIL_Phase_QSM Combine coil phase images and reconstruct QSM
%   

brainMask = load_nii([maskDir '/Brain_MASK.nii.gz']);
brainMask = brainMask.img>0;

phaseVol = [];

for i=0:31
    nii = load_nii([inDir '/TissuePhase_Coil_' num2str(i) '_1_PHASE.nii.gz']);
    mag = load_nii([rawDir '/Raw_Coil_' num2str(i) '_1_MAGNITUDE.nii.gz']);
    
    % Complex sum of coils weighted by magnitude
    complexVol = mag.img.*exp(1i*nii.img);
    if isempty(phaseVol)
        phaseVol = complexVol;
    else
        phaseVol = phaseVol + complexVol;
    end   
end

% Take phase of complex sum
phaseVol = angle(phaseVol);

% Calc params
params.H = [0 0 1];
params.voxelsize = nii.hdr.dime.pixdim(2:4);
params.padsize = [12 12 12];
%params.niter = max_niter;
%params.cropsize = cropsize;
params.TE = 20;
params.B0 = 3;
%params.tol_step1 = tol_step1;
%params.tol_step2 = tol_step2;
%params.Kthreshold = Kthreshold;

% Reconstruct QSM
[Susceptibility]= QSM_iLSQR(phaseVol,brainMask,'params',params);
nii.img = Susceptibility;

save_nii(nii,[outDir '/QSM_Combined_Phase.nii.gz']);

end

