#!/bin/bash

# This general script will call the selected scripts for your DWI data processing. In data_path, you have to set the path to your subjects folder (don't forget to adapt the indices). Then select the scripts you want to run by uncommenting them.

# General structure:
# StudyX > data
#	     > subject_folder: 1 folder per subject
#		> data: in this folder you put all your DICOM images  
#		> results: this folder is made by scripts
# 	 > Scripts: put all scripts here (except this one: prepro_all.sh)
#	 > prepro_all.sh



##################################################################################

data_path[0]="${PWD}/data/CON01T1"
#data_path[1]="${PWD}/data/CON02T1"
#data_path[2]="${PWD}/data/PAT01T1"
#data_path[3]="${PWD}/data/PAT02T1"
#data_path[4]="${PWD}/data/PAT03T1"
#data_path[5]="${PWD}/data/PAT04T1"
#data_path[6]="${PWD}/data/PAT05T1"
#data_path[7]="${PWD}/data/PAT06T1"

scripts_path="Scripts"

echo "*************************************************************"
echo "***             SINGLE-SHELL DTI HARDI PROCESSING          ***"
echo "***                USING MRtrix3, FSL (& AFNI)             ***"
echo "*************************************************************"

for subj_path in ${data_path[*]}
do
  echo "Processing:${subj_path}"
  date
 
####################################################################

  echo "***********************************************************"
  echo "*** 1. Convert DICOMs to NIfTI                          ***"
  echo "***********************************************************"
  ${scripts_path}/dicom2nii.sh ${subj_path}
  #DONE

  echo "***********************************************************"
  echo "*** 2. DWI distortion correction	                ***"
  echo "***********************************************************"
  ${scripts_path}/dwi_distcor.sh ${subj_path}
  #DONE

  echo "***********************************************************"
  echo "*** 3. Image registration to DWI space                  ***"
  echo "***********************************************************"
  ${scripts_path}/dwi_registration.sh ${subj_path}
  #DONE

 echo "***********************************************************"
  echo "*** 4. T1 data preprocessing                          ***"
  echo "***********************************************************"
  ${scripts_path}/dwi_anatomic_pre.sh ${subj_path}
  #DONE

  echo "***********************************************************"
  echo "*** 5. Single-shell CSD: Estimate individual RF		***"  
  echo "***********************************************************"
  ${scripts_path}/dwi_1shellCSD-RF.sh ${subj_path}



#TODO: 
#	- calculate group response function (dwi2response)
#	- calculate fodf (dwi2fod)
#	- tracking: either specific tracts or whole-brain (tckgen)

done
