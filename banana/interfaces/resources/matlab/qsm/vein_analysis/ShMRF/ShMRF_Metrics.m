function [ scores ] = ShMRF_Metrics( prediction, truth, mask )
%SHMRF_METRICS Summary of this function goes here
%   Detailed explanation goes here

if nargin<3
    mask = ones(size(prediction));
end

prediction = prediction.*mask;
truth = truth.*mask;

% Mask
VP = (prediction==1) & (mask==1);
NP = (prediction==0) & (mask==1);
V = (truth==1) & (mask==1);
N = (truth==0) & (mask==1);

A = nnz(mask==1);
dVP = imdilate(VP,ball(1));
dNP = imdilate(NP,ball(1));
dV = imdilate(V,ball(1));
dN = imdilate(N,ball(1));

scores.TP = nnz(V&VP);
scores.TN = nnz(N&NP);
scores.FP = nnz(N&VP);
scores.FN = nnz(V&NP);

scores.dTP = 0.5*(nnz(dV&VP)+nnz(V&dVP));
scores.dTN = 0.5*(nnz(dN&NP)+nnz(N&dNP));

scores.ACC = (scores.dTP+scores.dTN)/nnz(mask==1);

scores.SE = nnz(dVP&V)/nnz(V);
scores.SP = nnz(dNP&N)/nnz(N);

scores.PPV = nnz(dV&VP)/nnz(VP);
scores.NPV = nnz(dN&NP)/nnz(NP);

scores.DSS = 2*scores.dTP/(nnz(V)+nnz(VP));

scores.MCC = ((scores.TP*scores.TN)-(scores.FP*scores.FN))/sqrt(nnz(VP)*nnz(V)*nnz(N)*nnz(NP));
if isnan(scores.MCC)
    scores.MCC=0;
end
scores.AVD = abs(scores.FP-scores.FN)/nnz(V);

% Distance from boundary using X - erode(X) as boundary mask
predictionBoundary = prediction & (~imerode(prediction,ball(1)));
truthBoundary = truth & (~imerode(truth,ball(1)));
D1 = bwdist(predictionBoundary);
D2 = bwdist(truthBoundary);
D1 = D1(truthBoundary);
D2 = D2(predictionBoundary);
scores.MHD = 0.5*(mean(D1(:))+mean(D2(:)));


% Score
scores.Z = (1-scores.DSS) + scores.AVD/2 + scores.MHD/2;


end

