function BackgroundFieldRemoval(dataDir, outDir, maskFile)

addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/'))

mkdir(outDir);

inFiles = dir([dataDir '/*_Unwrapped_PHASE.nii.gz']);

mask = load_nii(maskFile);
mask = mask.img>0;

if numel(inFiles) == 0
	disp('Files not found');
else
	disp([int2str(numel(inFiles)) ' real files found'])
    
	for i=1:numel(inFiles)
		inNii = load_nii([dataDir '/' inFiles(i).name]);
        
        % Calc phase from cartesian
        outNii = inNii;
                
        % Calc voxel dimensions
        voxelsize = outNii.hdr.dime.pixdim(2:4);
        
        %Remove Background Field       
        [TissuePhase,NewMask]=V_SHARP(inNii.img,mask,'voxelsize',voxelsize);
        outNii.img = TissuePhase;
        
        % Generate filename1
        outFile = inFiles(i).name;
        suffixInd = strfind(outFile,'_Unwrapped_PHASE');
        outFile = [outDir '/' outFile(1:suffixInd-1) '_Local_PHASE.nii.gz'];
        
        % Save Unwrapped Phase
        save_nii(outNii,outFile);        
	end
	
end

