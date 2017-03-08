function Unwrap_Single_Echo(inDir, outDir, nCoils)
%UNWRAP_SINGLE_ECHO
% Apply Laplacian unwrapping from STI suite to each coil

for i=0:(nCoils-1)
    pha = load_nii([inDir '/Raw_Coil_' num2str(i) '_1_PHASE.nii.gz']);
    
    
    % Calc voxel dimensions
    voxelsize = pha.hdr.dime.pixdim(2:4);
    padsize=[12 12 12];
    
    %Unwrap
    [Unwrapped_Phase, ~]=MRPhaseUnwrap(pha.img,'voxelsize',voxelsize,'padsize',padsize);
    pha.img = Unwrapped_Phase;
    
    save_nii(pha,[outDir '/Unwrapped_Coil_' num2str(i) '_1_PHASE.nii.gz']);
end

end