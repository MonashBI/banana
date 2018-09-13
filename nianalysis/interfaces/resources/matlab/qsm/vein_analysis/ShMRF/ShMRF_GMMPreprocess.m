function [ outData ] = ShMRF_GMMPreprocess( inData, inMask, inVol, inHyperintense )
%SHMRF_GMMPREPROCESS Summary of this function goes here
%   Detailed explanation goes here

% remove boundaries for training
trainingMask = imerode(inMask(:),ball(2))==1;

% initial seed is top inVol% of the intensities
if inHyperintense
    seed = (2 - single(inData>prctile(inData(trainingMask),100*(1-inVol))));
else
    seed = (2 - single(inMask).*single(inData<prctile(inData(trainingMask),100*inVol)));
end
    

% fit gmm and calculate posterior
dataGM = fitgmdist(inData(trainingMask),2,'Start',seed(trainingMask));
dataP = posterior(dataGM,inData(inMask(:)==1));
dataP = dataP(:,1);

% Values are clamped below 50th percentile to avoid
% false assignment when pr(V) decays to zero slower than pr(N)
% this is necessary when segmenting with class size discrepancy
dataRangeLimit = posterior(dataGM,prctile(inData(inMask(:)==1),50));

if inHyperintense
    dataRangeMask = inData(inMask(:)==1)<=prctile(inData(inMask(:)==1),50);
else
    dataRangeMask = inData(inMask(:)==1)>=prctile(inData(inMask(:)==1),50);
end
dataP(dataRangeMask) = dataRangeLimit(1);

% Reshape posterior to volume
outData = zeros(size(inMask));
outData(inMask==1) = dataP;
   
% Set limits of 0.01 and 0.99 and Normalise distribution (remove severe bias) 
% (unchanged if thresh% approx 50% likelihood)
if prctile(outData(trainingMask),100*(1-inVol))<0.25
    outData = max(0.01,min(0.99,exp(log(outData)+log(0.5/prctile(outData(trainingMask),100*(1-inVol))))));
else
    outData = max(0.01,min(0.99,outData));
end
end

