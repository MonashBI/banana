function HIP_ChannelCombination(inDir, outDir, nCoils)
%UNWRAP_SINGLE_ECHO
% Apply Laplacian unwrapping from STI suite to each coil

if nargin<3
    nCoils=32;
end

hip = [];
sumMag = [];

for i=1:nCoils
    pha1 = load_untouch_nii([inDir '/Raw_Coil_' num2str(i-1) '_1_PHASE.nii.gz']);
    mag1 = load_untouch_nii([inDir '/Raw_Coil_' num2str(i-1) '_1_MAGNITUDE.nii.gz']);
    
    pha2 = load_untouch_nii([inDir '/Raw_Coil_' num2str(i-1) '_2_PHASE.nii.gz']);
    mag2 = load_untouch_nii([inDir '/Raw_Coil_' num2str(i-1) '_2_MAGNITUDE.nii.gz']);
    
    if isempty(hip)
        hip = zeros(size(pha1.img));
        sumMag = zeros(size(pha1.img));
    end
    
    hip = hip + squeeze(mag1.img.*mag2.img.*exp(-1i.*(pha1.img-pha2.img)));
    sumMag = sumMag + mag1.img.*mag2.img;
end

pha1.img = angle(hip);
mag1.img = abs(hip);

save_untouch_nii(pha1,[outDir '/Raw_PHASE.nii.gz']);
save_untouch_nii(mag1,[outDir '/Raw_MAGNITUDE.nii.gz']);

mag1.img = mag1.img./sumMag;
save_untouch_nii(mag1,[outDir '/Raw_Q.nii.gz']);
