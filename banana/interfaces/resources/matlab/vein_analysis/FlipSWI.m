function FlipSWI(inputFile, hdrFile, outputFile)

in = load_untouch_nii(inputFile);
hdr = load_untouch_nii(hdrFile);

in.img = flip(flip(in.img,2),1);
in.hdr = hdr.hdr;

save_untouch_nii(in, outputFile);

end