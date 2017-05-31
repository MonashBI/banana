function HIP_ChannelCombination(inDir, outDir, nCoils)
%UNWRAP_SINGLE_ECHO
% Apply Laplacian unwrapping from STI suite to each coil

if nargin<3
    nCoils=4;
end

pha = [];
mag = [];
for i=0:(nCoils-1)
    pNii = load_nii([inDir '/Raw_Coil_' num2str(i) '_1_PHASE.nii.gz']);
    mNii = load_nii([inDir '/Raw_Coil_' num2str(i) '_1_MAGNITUDE.nii.gz']);
    
    if isempty(pha)
        pha = zeros([size(pNii.img) nCoils 2]);
        mag = zeros([size(pNii.img) nCoils 2]);
    end
    
    pha(:,:,:,i,1) = pNii.img; %#ok<AGROW>
    mag(:,:,:,i,1) = mNii.img; %#ok<AGROW>
    
    pNii = load_nii(['Raw/Raw_Coil_' num2str(i) '_2_PHASE.nii.gz']);
    mNii = load_nii(['Raw/Raw_Coil_' num2str(i) '_2_MAGNITUDE.nii.gz']);
    
    pha(:,:,:,i,2) = pNii.img; %#ok<AGROW>
    mag(:,:,:,i,2) = mNii.img; %#ok<AGROW>
    
    hipInner = squeeze(mag(:,:,:,:,1).*mag(:,:,:,:,2).*exp(-1i.*(pha(:,:,:,:,1)-pha(:,:,:,:,2))));
    hip = sum(hipInner,4);
    
    pNii.img = angle(hip);
    mNii.img = abs(hip);
    
    save_nii(pNii,[outDir '/Raw_PHASE.nii.gz']);
    save_nii(mNii,[outDir '/Raw_MAGNITUDE.nii.gz']);
end