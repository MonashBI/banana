function [Img_recon_ch, smaps] = recon_grappa2(data_file, ref_file, hdr_file, out_mag_file, out_channels_dir, Rpe, smapFlag)
% RECON_SIEMENSDATFILE_GRAPPA2 
% This function reconstruct images from Siemens Raw data file (*.dat, VB/VD/VE versions)
% This function works for GRAPPA factor of 2, 3D imaging and mutiple echos
% optionally sensitiviy maps can be estimated 
% 
% [Img_recon_ch, smaps] = RECON_SIEMENSDATFILE_GRAPPA2(datFilePath, smapsFlag) 
% Input:
% datFilePath: path to raw data file (*.dat)
% smapFlag: 0 - do not compute sensitivity maps, 1 - compute sensitivity maps
% 
% Outputs:
% Img_recon_ch: complex valued individual channel images
% smaps : complex valued individual channels sensitivity maps if smapsFlag=1 else smaps returns zero
% header : Header information about scan

% Author: Kamlesh Pawar 
% Date: 2019/03/27 12:02:48 
% Revision: 0.1 $
% Institute: Monash Biomedical Imaging, Monash University, Australia, 2019

% Read header
hdr = jsondecode(fileread(hdr_file));
arr_size = [hdr.num_channels, hdr.dims(1), hdr.dims(2), hdr.dims(3), hdr.num_echos];

% Read data files
S = load(data_file, '-mat');
ksDataScan = S.data_scan;

S = load(ref_file, '-mat');
ksDataCalib = S.calib_scan;

% fid = fopen(data_file);
% ksDataScan = reshape(fread(fid), arr_size);
% fclose(fid);

% fid = fopen(ref_file);
% ksDataCalib = reshape(fread(fid), arr_size);
% fclose(fid);

[CH, FE, PE, PAR, ECHO] = size(ksDataScan);
I_PE = hdr.dims(2);
I_PAR = hdr.dims(3);
voxel_size = hdr.voxel_size

%  ---- Handle Partial Fourier ---- %

zpad_pe = I_PE - PE;
zpad_par = I_PAR - PAR;
ksDataScan = cat(3, zeros(CH, FE, zpad_pe, PAR, ECHO), ksDataScan);
ksDataScan = cat(4, zeros(CH, FE, I_PE, zpad_par, ECHO), ksDataScan);
[CH, FE, PE, PAR, ECHO] = size(ksDataScan);
ksDataCalib = cat(4, zeros(CH, FE, size(ksDataCalib,3), zpad_par, ECHO), ksDataCalib);

% ----- Compute kspace shift ----- %
[~, indx] = max(abs(ksDataScan(:)));
[~, indx_fe, indx_pe, indx_par, ~] = ind2sub(size(ksDataScan), indx);
ks_shift_pe = PE/2 - indx_pe;
ks_shift_fe = FE/2 - indx_fe;
ks_shift_par = PAR/2 - indx_par;
ksDataScan = circshift(ksDataScan, ks_shift_par, 4);
ksDataCalib = circshift(ksDataCalib, ks_shift_par, 4);

% ----- FFT in partition direction ----- %
ksDataCalib = fftshift(fft(ifftshift(ksDataCalib, 4), [], 4), 4);
ksDataScan = fftshift(fft(ifftshift(ksDataScan, 4), [], 4), 4);
ksDataCalib = ksDataCalib/max(abs(ksDataCalib(:)));
ksDataScan = ksDataScan/max(abs(ksDataScan(:)));

% ------- Grappa Recon Params ----- %
R       =   [1,Rpe];
kernel  =   [5,4];
mask = zeros(CH,FE,PE);
mask(:,:,Rpe:Rpe:end) = 1;
Img_recon_ch = zeros(CH, FE, PE, PAR, ECHO);
ksrecon_ch = zeros(CH, FE, PE, PAR, ECHO);

% ---- Grappa Recon ----- %
TOT_ITER = ECHO*PAR;
iter = 1;
for curr_echo = 1:ECHO
    for curr_par = 1:PAR
        fprintf('Completed GRAPPA Recon: %.2f %%\n', (iter-1)/TOT_ITER*100)
        data = squeeze(ksDataScan(:,:,:,curr_par, curr_echo)).*mask;
        calib = squeeze(ksDataCalib(:,:,:,curr_par, curr_echo));
        ksrecon = grappa(data, calib, R, kernel);
        ksrecon = circshift(ksrecon, ks_shift_pe, 3);
        ksrecon = circshift(ksrecon, ks_shift_fe, 2);
        Irecon = fftshift(fft(ifftshift(ksrecon, 2), [], 2), 2);
        Irecon = fftshift(fft(ifftshift(Irecon, 3), [], 3), 3);
        Img_recon_ch(:,:,:,curr_par, curr_echo) = Irecon;
        ksrecon_ch(:,:,:,curr_par, curr_echo) = ksrecon;
        iter = iter + 1;
    end
end
clear ksDataCalib ksDataScan

for i=1:size(Img_recon_ch,5)
    Img_recon_ch(:,:,:,:,i) = flip(flip(permute(Img_recon_ch(:,:,:,:,i),[1 3 2 4 5]),2),3);
end


% ----- Compute Sensitivity Maps ----- %
if smapFlag==1
    fprintf('Starting Sensitivity maps estimation\n')
    smaps = zeros(CH, FE, PE, PAR, ECHO);
    half_kmax = 64;
    ksrecon_ch_lowres = zeros(CH, FE, PE, PAR, ECHO);
    ksrecon_ch_lowres(:,FE/2-half_kmax:FE/2+half_kmax, PE/2-half_kmax:PE/2+half_kmax, :, :)...
        = ksrecon_ch(:,FE/2-half_kmax:FE/2+half_kmax, PE/2-half_kmax:PE/2+half_kmax, :, :);
    img_ch_lowres = fftshift(fft(ifftshift(ksrecon_ch_lowres, 2), [], 2), 2);
    img_ch_lowres = fftshift(fft(ifftshift(img_ch_lowres, 3), [], 3), 3);
    for curr_echo = 1:ECHO
        fprintf('Completed Sensitivity maps estimation: %.2f%%\n', (curr_echo-1)/ECHO*100)
        yn = double(squeeze(img_ch_lowres(:,:,:,:,curr_echo)));
        yn = permute(yn, [2, 3, 4, 1]);
        [~,smaps(:,:,:,:,curr_echo)] = adapt_array_3d(yn);    
    end
else
    smaps = NaN;
end

% Calculate combined magnitude, and real and imaginary images per
% channel and save to nifti files
mag = squeeze(sqrt(sum(sum(abs(Img_recon_ch).^2, 1), 5)));
out_nii = make_nii(mag, voxel_size, [], [], 'Sum of squares magnitude average across echos');
save_nii(out_nii, out_mag_file);

for i=1:size(Img_recon_ch, 1)
    coil = squeeze(Img_recon_ch(i, :, :, :, :));

    out_nii = make_nii(real(coil), voxel_size, [], [], 'Real image per coil');
    save_nii(out_nii, sprintf('%s%sreal_%d.nii.gz', out_channels_dir, filesep, i));

    out_nii = make_nii(imag(coil), voxel_size, [], [], 'Imaginary image per coil');
    save_nii(out_nii, sprintf('%s%simaginary_c%d.nii.gz', out_channels_dir, filesep, i));
end
    
end
