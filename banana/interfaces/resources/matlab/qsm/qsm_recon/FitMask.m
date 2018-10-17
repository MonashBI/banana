function FitMask(inputFile, initialMaskFile, outputFile)

%maskFile = 'test_suit_right_dentate_in_qsm.nii.gz';
%qsmFile = 'optibet_t1p04_params.nii.gz';
%addpath(genpath('/Users/philward/git/banana/banana/interfaces/resources/matlab/qsm'));
graphDisplay = false;

initialMask = load_untouch_nii(initialMaskFile);
smoothInitialMask = imdilate(initialMask.img,ball(3))>0;
regionSize = nnz(initialMask.img);

qsm = load_untouch_nii(inputFile);
smoothQSM = convn(qsm.img,fspecial3('gaussian',[7 7 7]),'same');
initialSeed = find(smoothInitialMask&(smoothQSM==max(smoothQSM(smoothInitialMask(:)))));

if graphDisplay
    [seedX, seedY, seedZ] = ind2sub(size(qsm.img),initialSeed); %#ok<UNRCH>
end

newMask = zeros(size(initialMask.img))>0;
newMask(initialSeed(1)) = 1;

areaGrowing = true;
initialGrowth = true;

meanArray = zeros(2000,1);
difArray = zeros(2000,1);
i = 0;
while areaGrowing && (i < (regionSize*3))
    dilMap = convn(newMask,ball(1),'same');
    adjMap = (dilMap>0)&~newMask;
    
    adjList = find(adjMap);
    adjWeight = dilMap(adjMap);
  
    if initialGrowth 
        % Initially favour hyperintensities
        adjDif = max(0,mean(smoothQSM(newMask))-smoothQSM(adjList));
        % Penalise neighbours with fewer adjacent foxels
        adjDif = adjDif./adjWeight;
    else
        adjDif = abs(mean(smoothQSM(newMask))-smoothQSM(adjList));
    end
    
    % Add 3 at a time
    for j=1:3
        i = i+1;
        
        [~,newVox] = min(adjDif);
        newMask(adjList(newVox(1))) = 1;
    
        meanArray(i) = mean(qsm.img(newMask));
        difArray(i) = adjDif(newVox(1))/meanArray(i);
    
        adjDif(newVox) = 99;
    end

    if graphDisplay
    
        fprintf('Mean = %0.5g, Diff(%%) = %0.5g \n', meanArray(i), difArray(i)); %#ok<UNRCH>

        subplot(2,2,1)
        plot(1:numel(meanArray),meanArray,1:numel(difArray),difArray)
        title(sprintf('Initial growth period = %d',initialGrowth));

        subplot(2,2,2)
        imagesc(ones(size(squeeze(qsm.img(seedX,:,:))))')
        hold on
        h = imagesc(squeeze(qsm.img(seedX,:,:))');
        hold off
        set(h,'AlphaData',1-0.5*squeeze(newMask(seedX,:,:))');
        axis xy
        axis image
        subplot(2,2,3)
        imagesc(ones(size(squeeze(qsm.img(:,seedY,:))))')
        hold on
        h = imagesc(squeeze(qsm.img(:,seedY,:))');
        hold off
        set(h,'AlphaData',1-squeeze(newMask(:,seedY,:))');
        axis xy
        axis image
        subplot(2,2,4)
        imagesc(ones(size(squeeze(qsm.img(:,:,seedZ))))')
        hold on
        h = imagesc(squeeze(qsm.img(:,:,seedZ))');
        hold off
        set(h,'AlphaData',1-squeeze(newMask(:,:,seedZ))');
        axis xy
        axis image
        colormap gray

        drawnow
    end
    
    if initialGrowth && nnz(newMask)>(regionSize*0.7)
        initialGrowth = false;
        %fprintf('Initial period over')
    end
    
    if ~initialGrowth
        linFit = fit((1:100)',(difArray((i-99):i)),'poly1');
        areaGrowing = linFit.p1 > 0;
    end
end

%newMask = imerode(newMask,ball(1));
%CC = bwconncomp(newMask>0);
%numOfPixels = cellfun(@numel,CC.PixelIdxList);
%[~, largestComponent] = max(numOfPixels);
%newMask(:) = 0;
%newMask(CC.PixelIdxList{largestComponent}) = 1;
%newMask = imdilate(newMask,ball(1));
initialMask.img = double(newMask);
save_untouch_nii(initialMask, outputFile);

end
   

