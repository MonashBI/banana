function ComplexSum(dataDir)

addpath(genpath('/data/project/Phil/ASPREE_QSM/scripts/Nifti'))

originalDir = pwd;
cd(dataDir)

reFiles = dir('./*REAL.nii.gz');
imFiles = dir('./*IMAGINARY.nii.gz');

if numel(reFiles)*numel(imFiles) == 0
	disp('Files not found');
else
	disp([int2str(numel(reFiles)) ' real files found'])
	disp([int2str(numel(imFiles)) ' imaginary files found'])
	reNii = load_nii(reFiles(1).name);
	imNii = load_nii(imFiles(1).name);
    
    save_nii(reNii,reFiles(1).name);
    save_nii(imNii,imFiles(1).name);

	outMag.hdr = reNii.hdr;
	outMag.img = (single(reNii.img).^2)+(single(imNii.img).^2);

	norm = sqrt((single(reNii.img).^2)+(single(imNii.img).^2));

	for i=2:numel(reFiles)
		reNii = load_nii(reFiles(i).name);
		imNii = load_nii(imFiles(i).name);
            
        save_nii(reNii,reFiles(i).name);
        save_nii(imNii,imFiles(i).name);

		outMag.img = outMag.img + (single(reNii.img).^2)+(single(imNii.img).^2);
		norm = norm + sqrt((single(reNii.img).^2)+(single(imNii.img).^2));
	end

	outMag.img = outMag.img./norm;

	outFile = reFiles(end).name;
	suffixInd = strfind(outFile,'_Coil');
	outFile = [outFile(1:suffixInd-1) '_CSum_Mag.nii.gz'];
	
	save_nii(outMag,outFile);
	
end

cd(originalDir);
end
