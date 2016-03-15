#!/bin/bash

# This script will convert your raw DICOM images to NIfTI format, using MRtrix3 "mrconvert". If you run the script, the series to be converted will pop up (either "T1 conversion" or "DWI conversion"). You then have to select the appropriate data from the list provided. Lastly, a summary is provided of your different DWI shells.

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
  
  # Create the folders to store results
  results="${path[$i]}/results" 
  mkdir -p ${results}

  # Convert DICOMs to NIfTI
  echo "T1 conversion"
  mrconvert ${path[$i]}/data/ ${results}/T1.nii -stride 1,2,3
  echo "DWI conversion"
  mrconvert ${path[$i]}/data/ ${results}/dwi_raw.nii -export_grad_fsl ${results}/bvecs ${results}/bvals -stride 1,2,3,4
  echo "DWI_PA conversion: press 'q' if no PA scan available"
  mrconvert ${path[$i]}/data/ ${results}/dwi_raw_PA.nii -stride 1,2,3,4
  echo "DWI shells"
  mrinfo ${path[$i]}/data/ -shells -shellcounts
  
done

