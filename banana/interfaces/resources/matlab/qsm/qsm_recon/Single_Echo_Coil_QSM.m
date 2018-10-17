function Single_Echo_Coil_QSM(dataDir)

% Add libraries for nifti and STI suite
addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/'))

% Prepare directory structure
rawDir = [dataDir '/Raw'];
maskDir = [dataDir '/Masks'];
unwrapDir = [dataDir '/Unwrapped'];
tissueDir = [dataDir '/TissuePhase'];
qsmDir = [dataDir '/QSM'];

if ~exist(unwrapDir,'dir')
    mkdir(unwrapDir);
end
if ~exist(tissueDir,'dir')
    mkdir(tissueDir);
end
if ~exist(qsmDir,'dir')
    mkdir(qsmDir);
end

% Step 1: Prepare masks
% Coil specific masks and whole brain mask
% Perform instensity correction on whole-brain magnitude image
Prepare_Masks(rawDir, maskDir);

% Step 2: Unwrap phase
% Apply Laplacian unwrapping
Unwrap_Single_Echo(rawDir, unwrapDir);

% Step 3: Remove background phase
% Apply V-Shape
Background_Phase_Removal(unwrapDir, maskDir, tissueDir);

% Step 4: Reconstruct QSM
% Execute iLSQR on each coil
Coil_QSM(tissueDir, maskDir, qsmDir);

% Step 5: Combine Coils
Combined_Coil_QSM_p50(qsmDir, maskDir);

end
   
   

