function Coil_QSM( inDir, maskDir, outDir, nCoils )
%COIL_QSM Summary of this function goes here
%   Detailed explanation goes here

for i=0:(nCoils-1)
    nii = load_nii([inDir '/TissuePhase_Coil_' num2str(i) '_1_PHASE.nii.gz']);
    mask = load_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);
    
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
    [Susceptibility]= QSM_iLSQR(nii.img,mask.img>0,'params',params);
    nii.img = Susceptibility;
    
    save_nii(nii,[outDir '/QSM_Coil_' num2str(i) '.nii.gz']);
end

end

