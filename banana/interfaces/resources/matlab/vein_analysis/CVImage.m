function CVImage(qsm, swi, vein_atlas, mask, q_prior, s_prior, a_prior, outputFile)

mask = load_untouch_nii(mask);
mask = single(mask.img)>0;

[ swi, qsm, vein_atlas, hdrInfo ] = Inputs_IO( mask, swi, qsm, vein_atlas);

s_prior = load_untouch_nii(s_prior);
q_prior = load_untouch_nii(q_prior); 
a_prior = load_untouch_nii(a_prior); 

s_prior = single(s_prior.img);
q_prior = single(q_prior.img);
a_prior = single(a_prior.img);

cvVol = swi.*s_prior + qsm.*q_prior + vein_atlas.*a_prior;
cvVol = cvVol./(s_prior+q_prior+a_prior);

cvNii = make_nii(cvVol.*mask);
cvNii.hdr = hdrInfo;
save_nii(cvNii,outputFile);


end