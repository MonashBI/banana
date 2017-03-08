function Combined_Coil_QSM_p50( qsmDir, tissueDir, maskFile, maskDir, nCoils)
%COMBINED_COIL_QSM Combine coil specific QSM
%   Using mean within mask, might be better to use median in future

brainMask = load_nii(maskFile);
brainMask = brainMask.img>0;

qsmVol = [];
missingValues = [];
dims = [];

for i=0:(nCoils-1)
    nii = load_nii([qsmDir '/QSM_Coil_' num2str(i) '.nii.gz']);
    mask = load_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);
    
    qsmVol(:,i+1) = nii.img(:).*(mask.img(:)>0)-99*(mask.img(:)==0);
    missingValues(:,i+1) = mask.img(:)==0;
    
    if isempty(dims)
        dims = size(nii.img);
    end
end

% Order values so median value is at index 16
qsmVol = sort(qsmVol,2);

% Adjust median index (16) based on missing values
indVol=sub2ind(size(qsmVol),1:size(qsmVol,1),nCoils-floor(0.5*(nCoils-sum(missingValues,2)')));

% Take median value using index, resize and mask out background
medVol=reshape(qsmVol(indVol),dims);
medVol(medVol==-99) = 0;
medVol(brainMask==0) = 0;

% Save output
nii.img = medVol;
save_nii(nii,[qsmDir '/QSM.nii.gz']);

% Store all coil phase and coil masks in single volumes
tissueVol = zeros([size(nii.img) nCoils]);
maskVol = zeros([size(nii.img) nCoils]);

for i=0:(nCoils-1)
    tissue = load_nii([tissueDir '/TissuePhase_Coil_' num2str(i) '_1_PHASE.nii.gz']);
    mask = load_nii([tissueDir '/TissueMask_Coil_' num2str(i) '_1_MASK.nii.gz']);
    
    tissueVol(:,:,:,i+1) = tissue.img;
    maskVol(:,:,:,i+1) = mask.img;
end

% Adjust header for extra dimension
nii.hdr.dime.dim(4) = nCoils;

% Save outputs
nii.img = tissueVol;
save_nii(nii,[tissueDir '/TissuePhase.nii.gz']);

nii.img = maskVol;
save_nii(nii,[tissueDir '/CoilMasks.nii.gz']);

end
