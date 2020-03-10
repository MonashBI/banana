function convert_twix(in_file, out_file, out_ref, out_hdr)

% Read Twix file
data_obj = mapVBVD(in_file, 'removeOS');

% Pick largest data object in file
if length(data_obj)>1
    multi_obj = data_obj;
    acq_length = cellfun(@(x) x.image.NAcq, multi_obj);
    [~,ind] = max(acq_length);
    data_obj = data_obj{ind};
end
header = data_obj.hdr;

data_obj

% Get data arrays
calib_scan = permute(data_obj.refscan{''}, [2, 1, 3, 4, 5]);
data_scan = permute(data_obj.image{''}, [2, 1, 3, 4, 5]);

% Get full dimensions from header
num_freq = data_obj.hdr.Config.NImageCols;
num_phase = data_obj.hdr.Config.NPeFTLen;
num_partitions = data_obj.hdr.Config.NImagePar;
dims = [num_freq, num_phase, num_partitions];

% Get channel and echo information from header
if isfield(header.Config,'RawCha') &&...
    ~isempty(header.Config.RawCha)
    num_channels = header.Config.RawCha;
else
    num_channels = size(data_scan, 1);
end
if isfield(header.Meas,'RawEcho')
    num_echos = header.Meas.RawEcho;
elseif isfield(header.MeasYaps,'lContrasts')
    num_echos = header.MeasYaps.lContrasts;
else
    num_echos = size(data_scan, 5);
end

% Get Voxel size
voxel_size = [0, 0, 0];
slice_array = header.Phoenix.sSliceArray.asSlice{1};
voxel_size(1) = slice_array.dReadoutFOV / num_freq;
voxel_size(2) = slice_array.dPhaseFOV / num_phase;
voxel_size(3) = slice_array.dThickness / num_partitions;

% Get other parameters
if isfield(header.Meas,'alTE')
    TE = header.Meas.alTE(1:num_echos) * 1E-6;
elseif isfield(header.MeasYaps,'alTE')
    TE = [header.MeasYaps.alTE{1:num_echos}] * 1E-6;
else
    disp('No header field for echo times');
    TE = 0.0;
end
B0_strength = header.Dicom.flMagneticFieldStrength;
B0_dir = [0 0 1];
larmor_freq = header.Dicom.lFrequency; % (Hz)

% Save Header values to JSON file
fid = fopen(out_hdr, 'w');
fprintf(fid, '{"dims": [%d, %d, %d], ', dims(1), dims(2), dims(3));
fprintf(fid, '"num_channels": %d, ', num_channels);
fprintf(fid, '"num_echos": %d, ', num_echos);
fprintf(fid, '"voxel_size": [%f, %f, %f], ', voxel_size(1), voxel_size(2), voxel_size(3));
fprintf(fid, '"B0_strength": %f, ', B0_strength);
fprintf(fid, '"B0_dir": [%f, %f, %f], ', B0_dir(1), B0_dir(2), B0_dir(3));
fprintf(fid, '"larmor_freq": %f, ', larmor_freq);
fprintf(fid, '"TE": [');
for echo_i=1:num_echos
    if echo_i ~= 1
        fprintf(fid, ', ');
    end
    fprintf(fid, '%f', TE(echo_i));
end
fprintf(fid, ']}');
fclose(fid);

% Save data and calibration scan to binary files
% fid = fopen(out_file, 'w');
% fwrite(fid, data_scan);
% fclose(fid);

save(out_file, 'data_scan', '-v7.3');

% fid = fopen(out_ref, 'w');
% fwrite(fid, calib_scan);
% fclose(fid);

save (out_ref, 'calib_scan', '-v7.3');
