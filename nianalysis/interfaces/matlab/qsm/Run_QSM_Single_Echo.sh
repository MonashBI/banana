#! /bin/bash

module load fsl
module load matlab

dataDir=$1
matlabDir=/data/project/Phil/ASPREE_QSM/scripts/

# Prepare data format and construct magnitude image
matlab -nosplash -nodisplay -r "addpath(genpath('$matlabDir')); Single_Echo_Coil_QSM('$dataDir') ; exit;"
