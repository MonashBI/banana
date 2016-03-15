#!/bin/bash

# This script will perform the 5 tissue type segmentation and creates a map of the grey matter-white matter interface needed for ACT.
# NOTE: adapt your path to the mrtrix3 scripts folder
# IMPORTANT: Check the quality of the gmwmi map by overlaying it on the T1_regis.nii image and check if both images match.

##################################################################################


argc=$#
argv=("$@")

ACT_path="/usr/local/mrtrix3/0.3.12/scripts"

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
  results="${path[$i]}/results" 
  # Make sure there is an anatomical
  t1_file="${results}/T1_regis.nii"
  if [ ! -f $t1_file ]; then
    echo "Error: ${results}/T1_regis.nii not found!"
    exit
  fi 


echo "*** Preprocessing anatomical images: 5tt segmentation ***"
  # Preparation for ACT: Tissue segmentation
  ${ACT_path}/act_anat_prepare_fsl ${results}/T1_regis.nii ${results}/5tt.nii
  5tt2gmwmi ${results}/5tt.nii ${results}/gmwmi.nii
  echo ">> Remember to check the quality of your gmwmi!"

done

