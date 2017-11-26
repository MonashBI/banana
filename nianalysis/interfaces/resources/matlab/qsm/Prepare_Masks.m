function Prepare_Masks( inDir, maskFile, outDir, nCoils, echoId)
%PREPARE_MASKS
%   Coil specific and whole-brain masks using FSL-BET

if nargin<5
    echoId=1;
end

% Spherical structure element
SE = fspecial3('ellipsoid',[11 11 11]);

% Load whole brain mask
mask = load_untouch_nii(maskFile);
mask = mask.img>0;
mask = imdilate(mask,SE>0);

%Generate coil specific masks by thresholding magnitude image
for i=0:(nCoils-1)
    inMag = load_untouch_nii([inDir '/Raw_Coil_' num2str(i) '_' num2str(echoId) '_MAGNITUDE.nii.gz']);
    
    % Blur to remove tissue based contrast
    outVol = convn(inMag.img,SE,'same');
    % Threshold to high-signal area
    outVol = outVol>graythresh(inMag.img);
    % Remove orphaned pixels and then close holes in mask    
    outVol = imclose(outVol,SE>0)>0;
    outVol = imopen(outVol,SE>0)>0;
    
    % Clip to brain mask region
    outMask = inMag;
    outMask.img = (outVol.*mask)>0;
    
    save_untouch_nii(outMask,[outDir '/Coil_' num2str(i) '_MASK.nii.gz']);
 
end

end
