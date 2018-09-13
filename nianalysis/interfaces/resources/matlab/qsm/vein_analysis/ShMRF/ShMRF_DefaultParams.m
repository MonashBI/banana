function [ params ] = ShMRF_DefaultParams(  )
%SHMRF_GENPARAMS Summary of this function goes here
%   Detailed explanation goes here

params.omega1 = 0.5;
params.omega2 = 0.5;
params.initialvol = 0.05;
params.scales = 0.5:0.5:2.5;
params.labelbright = true;
params.preprocess = true;
params.display = false;

end

