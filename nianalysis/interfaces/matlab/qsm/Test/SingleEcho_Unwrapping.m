function SingleEcho_Unwrapping(dataDir, outDir)

addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/'))

mkdir(outDir);

reFiles = dir([dataDir '/*REAL.nii.gz']);
imFiles = dir([dataDir '/*IMAGINARY.nii.gz']);

if numel(reFiles)*numel(imFiles) == 0
	disp('Files not found');
else
	disp([int2str(numel(reFiles)) ' real files found'])
	disp([int2str(numel(imFiles)) ' imaginary files found'])
    
	for i=1:numel(reFiles)
		reNii = load_nii([dataDir '/' reFiles(i).name]);
		imNii = load_nii([dataDir '/' imFiles(i).name]);
        
        % Calc phase from cartesian
        phaseNii = reNii;
        phaseNii.img = angle(complex(reNii.img,imNii.img));
        
        % Generate filename1
        outFile = reFiles(i).name;
        suffixInd = strfind(outFile,'_REAL');
        outFile = [outDir '/' outFile(1:suffixInd-1) '_Wrapped_PHASE.nii.gz'];
        
        % Save Phase
        save_nii(phaseNii,outFile);
        
        % Calc voxel dimensions
        voxelsize = phaseNii.hdr.dime.pixdim(2:4);
        padsize=[12 12 12];
        
        %Unwrap        
        [Unwrapped_Phase, Laplacian]=MRPhaseUnwrap(phaseNii.img,'voxelsize',voxelsize,'padsize',padsize);
        phaseNii.img = Unwrapped_Phase;
        
        % Generate filename1
        outFile = reFiles(i).name;
        suffixInd = strfind(outFile,'_REAL');
        outFile = [outDir '/' outFile(1:suffixInd-1) '_Unwrapped_PHASE.nii.gz'];
        
        % Save Unwrapped Phase
        save_nii(phaseNii,outFile);
        
	end
	
end

