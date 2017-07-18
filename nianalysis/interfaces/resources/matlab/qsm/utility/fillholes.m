function fillholes( inFile, outFile )
%FILLHOLES Summary of this function goes here
%   Detailed explanation goes here

% test input for debugging
%inFile = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil1.nii.gz';

nii = load_nii(inFile);

vol = nii.img==0;

CC=bwconncomp(vol);
numVoxels = cellfun(@numel,CC.PixelIdxList);
smallHoles = find(numVoxels~=max(numVoxels));

for i=1:numel(smallHoles)
    nii.img(CC.PixelIdxList{smallHoles(i)}) = 1;
end

save_nii(nii,outFile);

end

