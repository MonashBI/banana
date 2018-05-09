function QSM_Reconstruction(dataDir, outDir, maskFile)

addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/'))

mkdir(outDir);

inFiles = dir([dataDir '/*_Local_PHASE.nii.gz']);

mask = load_nii(maskFile);
mask = mask.img>0;

if numel(inFiles) == 0
	disp('Files not found');
else
	disp([int2str(numel(inFiles)) ' files found'])
    
	for i=1:numel(inFiles)
		inNii = load_nii([dataDir '/' inFiles(i).name]);
        
        % Set headers
        outNii = inNii;
                
        % Calc params
        params.H = [0 0 1];
        params.voxelsize = outNii.hdr.dime.pixdim(2:4);
        params.padsize = [12 12 12];
        %params.niter = max_niter;
        %params.cropsize = cropsize;
        params.TE = 20;
        params.B0 = 3;
        %params.tol_step1 = tol_step1;
        %params.tol_step2 = tol_step2;
        %params.Kthreshold = Kthreshold;
            
        % Reconstruct QSM     
        [Susceptibility]= QSM_iLSQR(inNii.img,mask,'params',params);
        outNii.img = Susceptibility;
        
        % Generate filename
        outFile = inFiles(i).name;
        suffixInd = strfind(outFile,'_Local_PHASE');
        outFile = [outDir '/' outFile(1:suffixInd-1) '_QSM.nii.gz'];
        
        % Save QSM
        save_nii(outNii,outFile);        
	end
end