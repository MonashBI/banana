function Coil_QSM( inDir, maskDir, outDir, te, nCoils, echoId )
%COIL_QSM Summary of this function goes here
%   Detailed explanation goes here
if nargin<6
    echoId=1;
    if nargin<5
        nCoils=32;
        if nargin<4
            te=20;
        end
    end
end

for i=0:(nCoils-1)
    nii = load_untouch_nii([inDir '/TissuePhase_Coil_' num2str(i) '_' num2str(echoId) '_PHASE.nii.gz']);
    mask = load_untouch_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);
    
     % Calc params
    params.H = [0 0 1];
    params.voxelsize = nii.hdr.dime.pixdim(2:4);
    params.padsize = [12 12 12];
    %params.niter = max_niter;
    %params.cropsize = cropsize;
    params.TE = te;
    params.B0 = 3;
    %params.tol_step1 = tol_step1;
    %params.tol_step2 = tol_step2;
    %params.Kthreshold = Kthreshold;

    % Reconstruct QSM     
    [Susceptibility]= QSM_iLSQR(nii.img,mask.img>0,'params',params);
    nii.img = Susceptibility;
    
    save_untouch_nii(nii,[outDir '/QSM_Coil_' num2str(i) '_' num2str(echoId) '.nii.gz']);
end

end

