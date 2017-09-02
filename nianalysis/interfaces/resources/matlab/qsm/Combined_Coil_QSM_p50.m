function Combined_Coil_QSM_p50( qsmDir, tissueDir, maskFile, maskDir, nCoils, nEchos)
%COMBINED_COIL_QSM Combine coil specific QSM
%   Using mean within mask, might be better to use median in future
if nargin<6
    nEchos=1;
    if nargin<5
        nCoils=32;
    end
end

brainMask = load_nii(maskFile);
brainMask = brainMask.img>0;

qsmVol = [];
missingValues = [];
dims = [];

for i=0:(nCoils-1)
    for j=1:nEchos
        nii = load_nii([qsmDir '/QSM_Coil_' num2str(i) '_' num2str(j) '.nii.gz']);
        mask = load_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);

        qsmVol(:,i+1+(j-1)*nCoils) = nii.img(:).*(mask.img(:)>0)-99*(mask.img(:)==0);
        missingValues(:,i+1+(j-1)*nCoils) = mask.img(:)==0;

        if isempty(dims)
            dims = size(nii.img);
        end
        
        disp(['Progress: ' num2str(i) ',' num2str(j) ])
    end
end

% Order values so median value is at index 16
qsmVol = sort(qsmVol,2);

% Adjust median index (16) based on missing values
indVol=sub2ind(size(qsmVol),1:size(qsmVol,1),nCoils*nEchos-floor(0.5*(nCoils*nEchos-sum(missingValues,2)')));

% Take median value using index, resize and mask out background
medVol=reshape(qsmVol(indVol),dims);
medVol(medVol==-99) = 0;
medVol(brainMask==0) = 0;

% Save output
nii.img = medVol;
save_nii(nii,[qsmDir '/QSM.nii.gz']);

% Store all coil phase and coil masks in single volumes
tissueVol = zeros([size(nii.img) nCoils*nEchos]);
maskVol = zeros([size(nii.img) nCoils*nEchos]);

for j=1:nEchos
    for i=0:(nCoils-1)
        tissue = load_nii([tissueDir '/TissuePhase_Coil_' num2str(i) '_' num2str(j) '_PHASE.nii.gz']);
        mask = load_nii([tissueDir '/TissueMask_Coil_' num2str(i) '_' num2str(j) '_MASK.nii.gz']);

        tissueVol(:,:,:,i+1+(j-1)*nCoils) = tissue.img;
        maskVol(:,:,:,i+1+(j-1)*nCoils) = mask.img;
    end
end

% Adjust header for extra dimension
nii.hdr.dime.dim(5) = nCoils*nEchos;

% Save outputs
nii.img = tissueVol;
save_nii(nii,[qsmDir '/TissuePhase.nii.gz']);

nii.img = maskVol;
save_nii(nii,[qsmDir '/PhaseMasks.nii.gz']);

end
