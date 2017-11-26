function Prepare_Raw_Channels(inDir, filename, echo_times, num_channels, outDir, outFile_firstEcho, outFile_lastEcho)
%PREPARE_RAW_CHANNELS
%   Takes all REAL and IMAGINARY pairs in current directory and prepares
%   them for Phase and QSM processing.
%
%   1. Existence of pairs is checked
%   2. Files are load/save cycled for formatting and rename for consistency
%   3. Magnitude and Phase components are produced
%   4. Coils are combined for a single magnitude image
if isempty(filename)
    filename = 'T2swi3d_ axial_p2_0.9_iso_COSMOS_Straight_Coil';
end

if isempty(outFile_firstEcho)
    outFile_firstEcho = [outDir 'Raw_MAGNITUDE_FirstEcho.nii.gz'];
end

if isempty(outFile_lastEcho)
    outFile_lastEcho = [outDir 'Raw_MAGNITUDE_LastEcho.nii.gz'];
end

% Check output directory exists
if ~exist(outDir,'dir')
    mkdir(outDir);
end

% Shuffle random number generator to add noise
rng('shuffle')

% Sort echo times to ensure longest echo is first
sumSqrMag_fe = [];
sumMag_fe = [];
sumSqrMag_le = [];
sumMag_le = [];
for i=1:numel(echo_times)
    for j=1:num_channels
        reFilename = sprintf([inDir filesep filename '_%d_%d_%s.nii.gz'], (j-1),i, 'REAL');
        imFilename = sprintf([inDir filesep filename '_%d_%d_%s.nii.gz'], (j-1),i, 'IMAGINARY');
        
        try
        	inRe = load_untouch_nii(reFilename);
        catch
            throw(MException('Prepare_Raw_Channels:MissingRealFile',...
                sprintf('Unable to find file for coil %d, echo %d: %s', (j-1), i, reFilename)));
        end
        
        try
            inIm = load_untouch_nii(imFilename);
        catch
            throw(MException('Prepare_Raw_Channels:MissingImaginaryFile',...
                sprintf('Unable to find file for coil %d, echo %d: %s', (j-1), i, imFilename)));
        end
        
        % Remove extreme values
        inRe.img(inRe.img==2048) = 0.02*rand(nnz(inRe.img==2048),1);
        inIm.img(inIm.img==2048) = 0.02*rand(nnz(inRe.img==2048),1);
    
        % Construct polar images
        complexVol = complex(inRe.img,inIm.img);
        outMag = inRe;
        outMag.img = abs(complexVol);
        outPha = inRe;
        outPha.img = angle(complexVol);
        save_untouch_nii(outMag, sprintf([outDir filesep 'Raw_Coil_%d_%d_%s.nii.gz'], (j-1),i, 'MAGNITUDE'));
        save_untouch_nii(outPha, sprintf([outDir filesep 'Raw_Coil_%d_%d_%s.nii.gz'], (j-1),i, 'PHASE'));
    
        % Calculate combined magnitude image from shortest echo only
        if i==1
            if isempty(sumSqrMag_fe)
                sumSqrMag_fe = outMag.img.^2;
                sumMag_fe = outMag.img;
            else
                sumSqrMag_fe = sumSqrMag_fe + outMag.img.^2;
                sumMag_fe = sumMag_fe + outMag.img;
            end
        end
        
        % Calculate combined magnitude image from longest echo only
        if i==numel(echo_times)
            if isempty(sumSqrMag_le)
                sumSqrMag_le = outMag.img.^2;
                sumMag_le = outMag.img;
            else
                sumSqrMag_le = sumSqrMag_le + outMag.img.^2;
                sumMag_le = sumMag_le + outMag.img;
            end
        end
    end
end

% Normalise magnitude weighted whole brain and save
if isempty(sumSqrMag_fe)
    ME = MException('Prepare_Raw_Channels:noRawFiles', ...
        'No input files found to prepare.');
    throw(ME);
else
    outMag.img = sumSqrMag_fe./sumMag_fe;
    outMag.img(isnan(outMag.img)) = 0;
    save_untouch_nii(outMag, outFile_firstEcho);
    
    outMag.img = sumSqrMag_le./sumMag_le;
    outMag.img(isnan(outMag.img)) = 0;
    save_untouch_nii(outMag, outFile_lastEcho);
end

end

