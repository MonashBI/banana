function Prepare_Raw_Channels(inDir)
%PREPARE_RAW_CHANNELS
%   Takes all REAL and IMAGINARY pairs in current directory and prepares
%   them for Phase and QSM processing.
%
%   1. Existence of pairs is checked
%   2. Files are load/save cycled for formatting and rename for consistency
%   3. Magnitude and Phase components are produced
%   4. Coils are combined for a single magnitude image

outDir = [pwd '/Raw/'];
outFile = [outDir 'Raw_MAGNITUDE.nii.gz'];

% Check output directory exists
if ~exist(outDir,'dir')
    mkdir(outDir);
end

% Find all real signal files
reFiles = dir([inDir '/*REAL.nii.gz']);

% Prepare to construct whole brain image
sumSqrMag = [];
sumMag = [];

disp(['Number of files = ' numel(reFiles)])
disp(['Directory searched = ' inDir])

% Loop over channels/coils
for i=1:numel(reFiles)
    imFile = [inDir '/' reFiles(i).name(1:(end-length('REAL.nii.gz'))) 'IMAGINARY.nii.gz'];
    
    % Check imaginary exists
    if ~exist(imFile,'file')
        disp(['No imaginary signal for ' reFiles(i).name])
        disp('Process Terminated')
        exit;
    end
    
    % Load and save files for format consistency (and remove 2048 artefact)
    % Rename all files
    suffixInd = strfind(reFiles(i).name,'_Coil');
    fileName = [outDir 'Raw' reFiles(i).name(suffixInd:(end-length('_REAL.nii.gz')))];
    
    inRe = load_nii([inDir '/' reFiles(i).name]);
    inIm = load_nii([imFile]);
    inRe.img(inRe.img==2048) = 0;
    inIm.img(inIm.img==2048) = 0;
    save_nii(inRe, [fileName '_REAL.nii.gz']);
    save_nii(inIm, [fileName '_IMAGINARY.nii.gz']);
    
    % Constructave polar images
    complexVol = complex(inRe.img,inIm.img);
    outMag = inRe;
    outMag.img = abs(complexVol);
    outPha = inRe;
    outPha.img = angle(complexVol);
    save_nii(outMag, [fileName '_MAGNITUDE.nii.gz']);
    save_nii(outPha, [fileName '_PHASE.nii.gz']);
    
    % Accumulate whole brain image
    if isempty(sumSqrMag)
        sumSqrMag = outMag.img.^2;
        sumMag = outMag.img;
    else
        sumSqrMag = sumSqrMag + outMag.img.^2;
        sumMag = sumMag + outMag.img;
    end
end

% Normalise magnitude weighted whole brain and save
outMag.img = sumSqrMag./sumMag;
outMag.img(isnan(outMag.img)) = 0;
save_nii(outMag, outFile);

end

