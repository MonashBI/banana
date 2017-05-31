function QSM_DualEcho( inDir, maskFile, outDir )

% Add libraries for nifti and STI suite
addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/'))

% Prepare directory structure
mkdir([outDir '/QSM']);
phaseFile = [outDir '/QSM/Raw_PHASE.nii.gz'];
newMaskFile = [outDir '/QSM/PhaseMask.nii.gz'];
unwrapFile = [outDir '/QSM/Unwrapped.nii.gz'];
tissueFile = [outDir '/QSM/TissuePhase.nii.gz'];
qsmFile = [outDir '/QSM/QSM.nii.gz'];
nCoils = 32;

% Combine channels
HIP_ChannelCombination(inDir, [outDir '/QSM'], nCoils);

% Load Inputs (Raw phase and mask)
mask = load_nii(maskFile);
nii = load_nii(phaseFile);

% Calc params
params.H = [0 0 1];
params.voxelsize = nii.hdr.dime.pixdim(2:4);
params.padsize = [12 12 12];
params.TE = 14.76; % (22.14-7.38)
params.B0 = 3;

% Step 1: Unwrap phase
% Apply Laplacian unwrapping
[Unwrapped_Phase, ~]=MRPhaseUnwrap(nii.img,'voxelsize',params.voxelsize,'padsize',params.padsize);

% Save Intermediate Results
nii.img = Unwrapped_Phase;
save_nii(nii,unwrapFile);

% Step 2: Remove background phase
% Apply V-Shape             
[TissuePhase,NewMask]=V_SHARP(Unwrapped_Phase,mask.img>0,'voxelsize',params.voxelsize);

% Save Intermediate Results
nii.img = TissuePhase;
save_nii(nii,tissueFile);
nii.img = NewMask<0;
save_nii(nii,newMaskFile);

% Step 3: Reconstruct QSM
[Susceptibility]= QSM_iLSQR(TissuePhase,NewMask>0,'params',params);
nii.img = Susceptibility;

% Save QSM
save_nii(nii,qsmFile);

end
   
   

