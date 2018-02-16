function Background_Phase_Removal( inDir, maskDir, outDir, nCoils, echoId)
%BACKGROUND_PHASE_REMOVAL Apply V-Sharp to remove the background field
%   
if nargin<5
    echoId=1;
    if nargin<4
        nCoils=32;
    end
end

for i=0:(nCoils-1)
    nii = load_untouch_nii([inDir '/Unwrapped_Coil_' num2str(i) '_' num2str(echoId) '_PHASE.nii.gz']);
    mask = load_untouch_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);
    
    % Calc voxel dimensions
    voxelsize = nii.hdr.dime.pixdim(2:4);
        
    %Remove Background Field       
    [TissuePhase,NewMask]=V_SHARP(nii.img,mask.img>0,'voxelsize',voxelsize);
    nii.img = TissuePhase;
    
    save_untouch_nii(nii,[outDir '/TissuePhase_Coil_' num2str(i) '_' num2str(echoId) '_PHASE.nii.gz']);
    nii.img = NewMask<0;
    save_untouch_nii(nii,[outDir '/TissueMask_Coil_' num2str(i) '_' num2str(echoId) '_MASK.nii.gz']);
end

