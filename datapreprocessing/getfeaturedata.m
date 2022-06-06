function a = getfeaturedata(file_path)
% You need download BioSigKit first
% About BioSigKit https://joss.theoj.org/papers/10.21105/joss.00671
% The place to download the BioSigKit: https://github.com/hooman650/BioSigKit/tree/BioSigKitV1.1
% Has used some code and algorithms from :https://github.com/zzklove3344/ApneaECGAnalysis/tree/master/preprocessOfApneaECG
addpath(genpath('D:\#D\datapreprocessing\BioSigKit'))
read_path = strcat(file_path, '\denoised_ecg_data.mat');
ecg_data = load(read_path);
ecg_data_1 = ecg_data.denoised_ecg_data;
a = 0;

Analysis = RunBioSigKit(ecg_data_1, 100, 0);
try
    Analysis.MTEO_qrstAlg;
    Rwave = Analysis.Results.R;
    save_path = strcat(file_path, '\Rwave.mat');
    save(save_path, 'Rwave');
    disp('Successfully.');
catch
    disp('it is noise.');
end

