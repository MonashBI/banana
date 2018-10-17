function ShMRF(in_file, mask_file, out_file)

I = load_untouch_nii(in_file);
mask = load_untouch_nii(mask_file);
out = I;

mask = mask.img>0;
I = I.img;
I(isnan(I)) = 0;

%%
params = ShMRF_DefaultParams();
params.omega1 = 0.01;
params.omega2 = 0.20; %(0.12 in first round)
params.display = false;
params.preprocess = false;

out.img = ShMRF_Segment(I, mask, params);

save_untouch_nii(out, out_file);

end
