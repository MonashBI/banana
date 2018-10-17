function Dual_Echo_Coil_QSM(dataDir)

% Sequence params
nCoils = 32;
nEchos = 2;
te = [7.38 22.14];

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

% Step 1: Prepare masks (from first echo only)
% Coil specific masks and whole brain mask
% Perform instensity correction on whole-brain magnitude image
Prepare_Masks(rawDir, [maskDir '/Mask.nii.gz'], maskDir, nCoils, 1);

for i=1:nEchos
    % Step 2: Unwrap phase
    % Apply Laplacian unwrapping
    Unwrap_Single_Echo(rawDir, unwrapDir, 32, i);
    
    % Step 3: Remove background phase
    % Apply V-Shape
    Background_Phase_Removal(unwrapDir, maskDir, tissueDir, nCoils, i);
    
    % Step 4: Reconstruct QSM
    % Execute iLSQR on each coil
    Coil_QSM(tissueDir, maskDir, qsmDir, te(i), nCoils, i);
end

% Step 5: Combine Coils
Combined_Coil_QSM_p50(qsmDir, tissueDir, [maskDir '/Mask.nii.gz'], maskDir, nCoils, nEchos);

end