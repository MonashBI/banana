#! /bin/bash

module load fsl
module load matlab

dataDir=$1
reconDir=$dataDir/Recon
rawDir=$dataDir/Raw
maskDir=$dataDir/Masks
matlabDir=/data/project/Phil/ASPREE_QSM/scripts/

# Prepare data format and construct magnitude image
matlab -nosplash -nodisplay -r "addpath(genpath('$matlabDir')); Prepare_Raw_Channels('$reconDir','$rawDir') ; exit;"

# Calculate masks
mkdir $maskDir
bet $rawDir/Raw_MAGNITUDE.nii.gz $rawDir/BET_MAGNITUDE.nii.gz -B -f 0.3 -m
mv $rawDir/BET_MAGNITUDE_mask.nii.gz $maskDir/Brain_MASK.nii.gz
