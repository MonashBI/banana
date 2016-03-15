#!/bin/bash

# This script will create a mask for your DWI data, estimate the response function and the fibre orientation distribution function. 
# IMPORTANT: Check the quality of the mask & response function (shview response.txt). The response function should be broadest in the axial plane, and have low amplitude along the z-axis (see http://jdtournier.github.io/mrtrix-0.2/tractography/preprocess.html).


##################################################################################

argc=$#
argv=("$@")

# Check parameters
num_path=0
for (( i = 0; i<argc; i++ ))
do
 param=${argv[$i]}
 #if the user ask for help print help
 if [ "$param" = "-help" ] || [ "$param" = "-h" ]; then
    help_
 else
    path[$num_path]="$param"
    num_path=$num_path+1  
 fi
done

if [ "$num_path" = "0" ]; then
   echo "Error: No input_folder specify"
   help_
fi

##################################################################################

for (( i = 0; i<num_path; i++ ))
do
 
echo "*** Checking files & folders ***"
  # Create the folders to store results
  results="${path[$i]}/results" 


echo "*** Create DWI brain mask ***"
  bet ${results}/dwi_distcor.nii ${results}/dwi_mask.nii -f 0.2 -g 0.2 -n -m
  mv ${results}/dwi_mask_mask.nii.gz ${results}/dwi_mask.nii.gz  
  gunzip ${results}/dwi_mask.nii.gz
  echo ">> Remember to check the quality of your mask!"


echo "*** Response function estimation ***"
  dwi2response ${results}/dwi_distcor.nii ${results}/response.txt -fslgrad ${results}/bvecs ${results}/bvals -mask ${results}/dwi_mask.nii
  echo ">> Remember to check the quality of your response function!"

done  
