#!/bin/bash

# This script will register your anatomical data to the individual diffusion space, by transforming the transform matrix in the T1 header, thereby preserving the spatial resolution of the anatomical images.
# IMPORTANT: Check the registration quality by overlaying the T1_regis.nii on the dwi_distcorr.nii and check if both images match.


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


echo "*** Register T1 to diffusion space ***"
  flirt -in ${results}/T1.nii -ref ${results}/dwi_distcor.nii -dof 6 -cost mutualinfo -omat ${results}/anatomy2dwi_fsl.mat 
  transformcalc ${results}/anatomy2dwi_fsl.mat ${results}/anatomy2dwi_mrtrix.mat -flirt_import ${results}/T1.nii ${results}/dwi_distcor.nii
  mrtransform ${results}/T1.nii -linear ${results}/anatomy2dwi_mrtrix.mat -fslgrad ${results}/bvecs ${results}/bvals ${results}/T1_regis.nii
  echo ">> Remember to check the quality of your registration!"

done 

