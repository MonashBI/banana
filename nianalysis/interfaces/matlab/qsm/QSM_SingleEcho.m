function QSM_SingleEcho( inDir, maskFile, outDir )
%COIL_QSM Summary of this function goes here
%   Detailed explanation goes here

rawDir = inDir;
maskDir = [outDir '/Masks/'];
unwrapDir = [outDir '/Unwrapped/'];
tissueDir = [outDir '/TissuePhase/'];
qsmDir = [outDir '/QSM/'];
nCoils = 2;

% Check output directories exists
dirList = {maskDir, unwrapDir, tissueDir, qsmDir};
for i=1:numel(dirList)
	if ~exist(dirList{i},'dir')
    	mkdir(dirList{i});
	end
end

% Step 1: Prepare masks
% Coil specific masks and whole brain mask
% Perform instensity correction on whole-brain magnitude image
Prepare_Masks(rawDir, maskFile, maskDir, nCoils);

% Step 2: Unwrap phase
% Apply Laplacian unwrapping
Unwrap_Single_Echo(rawDir, unwrapDir, nCoils);

% Step 3: Remove background phase
% Apply V-Shape
Background_Phase_Removal(unwrapDir, maskDir, tissueDir, nCoils);

% Step 4: Invert Field
% Execute iLSQR on each coil
Coil_QSM(tissueDir, maskDir, qsmDir, nCoils);

% Step 5: Combine Coils
Combined_Coil_QSM_p50(qsmDir, tissueDir, maskFile, maskDir, nCoils);


end