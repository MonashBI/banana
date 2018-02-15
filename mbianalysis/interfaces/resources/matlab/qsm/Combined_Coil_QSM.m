function Combined_Coil_QSM( qsmDir, maskDir )
%COMBINED_COIL_QSM Combine coil specific QSM
%   Using mean within mask, might be better to use median in future

brainMask = load_nii([maskDir '/Brain_MASK.nii.gz']);
brainMask = brainMask.img>0;

qsmVol = [];
maskVol = [];

for i=0:31
    nii = load_nii([qsmDir '/QSM_Coil_' num2str(i) '.nii.gz']);
    mask = load_nii([maskDir '/Coil_' num2str(i) '_MASK.nii.gz']);
    
    if isempty(qsmVol)
        qsmVol = nii.img.*(mask.img>0);
        maskVol = mask.img;
    else
        qsmVol = qsmVol + nii.img.*(mask.img>0);
        maskVol = maskVol + mask.img;
    end
end

qsmVol = qsmVol./maskVol;
qsmVol(maskVol==0) = 0;
nii.img = qsmVol.*brainMask;

save_nii(nii,[qsmDir '/QSM.nii.gz']);

end

