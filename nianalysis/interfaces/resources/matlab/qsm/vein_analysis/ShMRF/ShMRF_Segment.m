function [ outLabels ] = ShMRF_Segment( inData, inMask, inParams )
%SHMRF_SEGMENT - Run Shape-based Markov Random Field segmentation method
%   Inputs
%       inData      - Input volume for processing
%       inMask      - Mask of region to perform segmentation within (must
%                       be same dimensions as inData)
%       inParams    - Options for processing (See ShMRF_DefaultParams.m for
%                       details explanation)
%   Outputs
%       outLabels   - Output volume of labels from ShMRF algorithms
%
%
%   Citation for use:
%       Ward, P.G.D, Ferris, N.F, Raniga, P., Ng, A.C.L., Barnes, D.G.,
%       Dowe, D.L., Egan G.F., 2017. Vein segmentatiom using Shape-based 
%       Markov Random Fields, in: 2017 IEEE 14th International Symposium on 
%       Biomedical Imaging (ISBI). Presented at the 2017 IEEE 14th
%       International Symposium on Biomedical Imaging (ISBI), pp.
%       1133-1136. doi:10.1109/ISBI.2017.XXXXXXX
%

% Cast to floating point
if ~isa(inData,'double') && ~isa(inData,'single')
    inData = single(inData);
end

if inParams.preprocess
    data = ShMRF_GMMPreprocess(inData, inMask, inParams.initialvol, inParams.labelbright);
else
    if ~inParams.labelbright
        data = max(inData(:))-inData;
    else
        data = inData;
    end
end


mask = inMask;
params(1) = inParams.omega1;
params(2) = inParams.omega2;
params(3) = 1-inParams.initialvol;
display=inParams.display;

if nargin<4
  % % calc hessian norm
    options.FrangiScaleRange = inParams.scales;
    options.FrangiScaleRatio = 0.5;
    options.FrangiAlpha = 0.5;
    options.FrangiBeta = 0.5;
    options.FrangiC = 0.01;
    options.BlackWhite = false;
    [~,~,Voutx,Vouty,Voutz]=FrangiFilter3D(data,options);  
end

BetaT = params(1);
BetaV = params(2);

%graphPos = zeros(nnz(mask), 14);
%graphNeg = zeros(nnz(mask), 14);


% % 
% % 

[cX,cY,cZ] = ndgrid(-1:1,-1:1,-1:1);
cX = cX(:);
cY = cY(:);
cZ = cZ(:);

% Remove local/self reference
cX(14) = [];
cY(14) = [];
cZ(14) = [];

% Generate vectors
vX = cX./sqrt(cX.^2+cY.^2+cZ.^2);
vY = cY./sqrt(cX.^2+cY.^2+cZ.^2);
vZ = cZ./sqrt(cX.^2+cY.^2+cZ.^2);
vX(isnan(vX)) = 0;
vY(isnan(vY)) = 0;
vZ(isnan(vZ)) = 0;

cliqueInd = sub2ind(size(mask),cX+3,cY+3,cZ+3);
cliqueInd = cliqueInd - sub2ind(size(mask),3,3,3);

volInd = repmat(cliqueInd,[1 numel(mask)])+repmat(1:numel(mask),[numel(cliqueInd) 1]);
volInd(volInd<=0) = 1;
volInd(volInd>numel(mask)) = 1;

Voutx(1) = 0;
Vouty(1) = 0;
Voutz(1) = 0;

vMax = abs(repmat(Voutx(:)',[26 1]).*repmat(vX(:),[1 numel(mask)]) + ...
    repmat(Vouty(:)',[26 1]).*repmat(vY(:),[1 numel(mask)]) + ...
    repmat(Voutz(:)',[26 1]).*repmat(vZ(:),[1 numel(mask)])).^2;% + ...
    %abs(Voutx(volInd).*repmat(vX(:),[1 numel(mask)]) + ...
    %Vouty(volInd).*repmat(vY(:),[1 numel(mask)]) + ...
    %Voutz(volInd).*repmat(vZ(:),[1 numel(mask)]));

cliqueWeights = vMax'.*repmat(26./sum(vMax,1)',[1 size(vMax,1)]);

% Labels
x = data(:)>0.5;
frozen=zeros(size(x));

%log p(X=x|Y=y) ~ log p(Y=y|X=x) + log p(X=x)
% Data term
% log p(Y=y | X=x) = log ( data(:,X=x) )
e1Data = log2(1-data(:));
e0Data = log2(data(:));

%1by1
maskInd = [];
%maskInd = find(mask(:) & ~frozen);
%maskInd = maskInd(randperm(numel(maskInd)));
%if display
%    figure('units','normalized','outerposition',[.67 0 .33 1])
%end

fCount=0;
i=0;
nTarget=nnz(mask);
nTask=numel(maskInd);
maxCycles = 20;
nCycles = 0;
labelVol = zeros(size(mask));
while nCycles<maxCycles

    i=i+1;
    if (i>numel(maskInd))
        maskInd=find(mask(:) & ~frozen);
        maskInd=maskInd(randperm(numel(maskInd)));
        nTask=numel(maskInd);
        i = 1;
        nCycles = nCycles+1;
        
        if display
            labelVol(:)=x;
            subplot(3,1,1)
            imagesc(labelVol(:,:,40)')
            axis image
            axis xy
            title(['Frozen = ' num2str(100*nnz(frozen)/nnz(mask)) '%']);
            subplot(3,1,2)
            imagesc(squeeze(labelVol(:,128,:))')
            axis image
            axis xy
            subplot(3,1,3)
            imagesc(squeeze(labelVol(112,:,:))')
            axis image
            axis xy
            drawnow
        end
    end
    
    % Random voxel to toggle
    clique = cliqueInd+maskInd(i);
    
    weights = cliqueWeights(maskInd(i),:);
    weights(clique<=0) = [];
    clique(clique<=0) = [];
    weights(clique>numel(mask)) = [];
    clique(clique>numel(mask)) = [];
    
    % Neighbourhood term
    % log p(X=x) = U(x) = sum(beta*(graphPos==x))
    % beta = neighbourhood potential
    % l(X|B) = sum_s log( e^-U(xs) / (e-U(xs=V) + e-U(xs=N)) )
    %e1Shape = exp(-BetaV*sum((x(clique)==1).*weights'));
    %e0Shape = exp(-BetaT*sum((x(clique)==0).*weights'));
    %e1Shape = e1Shape/(e1Shape + e0Shape); % Denominator
    %e0Shape = (1-e1Shape);
    e1Shape = BetaT*sum((x(clique)==0));
    e0Shape = BetaV*sum((x(clique)==1).*weights');
    
    newX = (e1Data(maskInd(i))+e1Shape)<(e0Data(maskInd(i))+e0Shape);

    if newX~=x(maskInd(i))
        if display
            %disp(['(' num2str(i) '/' num2str(nTask) ') Change ' num2str(x(maskInd(i))) ' -> ' num2str(newX)])
        end
        x(maskInd(i)) = newX;
        changeMask=x(clique)~=newX;
        fCount=fCount-nnz(frozen(clique(changeMask)));
        frozen(clique(changeMask)) = 0;
    end
    frozen(maskInd(i)) = 1;
    fCount=fCount+1;
        
    if fCount==nTarget
        break
    end
end

outLabels = false(size(inData));
outLabels(mask(:)==1) = x(mask(:)==1);

end

