#!/bin/bash

# This script will perform motion and eddy current (if you have a reverse phase encoding polarity image) correction of your DWI data.
# NOTE: Adapt the path to your revpe_distcorr script in the mrtrix3 folder
# IMPORTANT: If you use FSL EDDY, check the quality of the DWI brain mask by overlaying it on the raw DWI images and check if the entire brain is included.


##################################################################################

argc=$#
argv=("$@")

distcorr_path="/usr/local/mrtrix3/0.3.12/scripts"

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

  # Make sure there is a dwi file
  dwi_file="${results}/dwi_raw.nii"
  if [ ! -f $dti_file ]; then
    echo "Error: ${results}/dwi_raw.nii not found!"
    exit
  fi
  # Check whether there is a dti_PA
  dwirev_file="${results}/dwi_raw_PA.nii"
  # Make sure there is a bvals
  bval_file="${results}/bvals"
  if [ ! -f $bval_file ]; then
    echo "Error: ${results}/bvals not found!"
    exit
  fi
  # Make sure there is a bvecs
  bvec_file="${results}/bvecs"
  if [ ! -f $bvec_file ]; then
    echo "Error: ${results}/bvecs not found!"
    exit
  fi


echo "*** Distortion correction DWI images ***"

  if [ ! -f $dwirev_file ]; then
    echo ">> No reverse phase encoding polarity images found. I will use FSL EDDY!"

  	# Construct a configuration file
  	echo -e "0  1 0 0.1\n0 -1 0 0.1\n" > ${results}/config.txt

  	# Create mask for eddy
  	mrconvert ${results}/dwi_raw.nii -fslgrad ${results}/bvecs ${results}/bvals - | dwi2mask - - | maskfilter - dilate - | mrconvert - ${results}/dwi_mask_raw.nii -datatype float32
	echo ">> Remember to check the quality of your mask!"

  	# In case mask is not good, use FSL BET
  	#dwiextract -bzero ${results}/dwi_raw.nii -fslgrad ${results}/bvecs ${results}/bvals - | mrmath - -axis 3 mean ${results}/dwi_b0.nii
  	#bet ${results}/dwi_b0.nii ${results}/dwi_mask_raw.nii -f 0.4 -g 0.2 -n -m
  	#mv ${results}/dwi_mask_raw_mask.nii.gz ${results}/dwi_mask_raw.nii.gz
  	#gunzip ${results}/dwi_mask_raw.nii.gz
  	#maskfilter ${results}/dwi_mask_raw.nii dilate -npass 5 ${results}/dwi_mask_raw.nii

  	# Generate configuration file for eddy - index referring to PE and bandwidth for each volume. In this particular case, we assume that every volume in the series has the same imaging parameters as the first of the reversed-PE pair. Therefore, every volume has an index of 1.
  	num_volumes=$( wc -w < ${results}/bvals )
  	start=1
  	end=$num_volumes

  	indx=""
  	for ((i=$start; i<=$end; i+=1)); do indx="$indx 1"; done
  	echo $indx > ${results}/index.txt

  	# Run eddy
  	eddy --imain=${results}/dwi_raw.nii --mask=${results}/dwi_mask_raw.nii --index=${results}/index.txt --acqp=${results}/config.txt --bvecs=${results}/bvecs --bvals=${results}/bvals --out=${results}/dwi_distcor

  	# Reorganize results
  	eddy_results="$results/eddy"
  	mkdir -p $eddy_results
  	mv ${results}/dwi_distcor.eddy* ${eddy_results}
  	mv ${results}/config.txt ${eddy_results}
  	mv ${results}/index.txt ${eddy_results}
  	gunzip ${results}/dwi_distcor.nii.gz
  	cp ${eddy_results}/dwi_distcor.eddy_rotated_bvecs ${results}/bvecs

  else 
    echo ">> Found the reverse phase encoding polarity images! I will use revpe_distcorr!"
	dwiextract -bzero ${results}/dwi_raw.nii -fslgrad ${results}/bvecs ${results}/bvals - | mrmath - -axis 3 mean ${results}/b0_AP.nii
  	mrmath ${results}/dwi_raw_PA.nii -axis 3 mean ${results}/b0_PA.nii  
  	${distcorr_path}/revpe_distcorr ap ${results}/b0_AP.nii ${results}/b0_PA.nii ${results}/dwi_raw.nii ${results}/dwi_distcor.nii
  fi

done
