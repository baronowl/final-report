#To run this file, you need install matlab engine to the python first
import pywt
import os
import numpy as np
import matlab.engine
import scipy.io as sio
from scipy import interpolate
from datapreprocessing.operate import ECG_operate
from scipy.signal import decimate

#Has used some code and algorithms from :https://github.com/zzklove3344/ApneaECGAnalysis/tree/master/preprocessOfApneaECG

eng = matlab.engine.start_matlab()
ECGFrequency = 100
Base_path = "D:/#D/ecg-separate/"



def create_time_info(rri):
	rri_time = np.cumsum(rri) / 1000.0
	return rri_time - rri_time[0]

def create_interp_time(rri, fs):
	time_rri = create_time_info(rri)
	start, end = 0, 0
	if time_rri[-1] < 60:
		end = 60
	else:
		print("abnormal %s..." % time_rri[-1])
	return np.arange(0, end, 1 / float(fs))


def load_data(database_name, rdf, numRead=-1, is_debug=False):
	if database_name == ["apnea-ecg", "train"]:
		base_floder_path = Base_path + "train/"
	elif database_name == ["apnea-ecg", "test"]:
		base_floder_path = Base_path + "test/"
	else:
		raise Exception("Error command.")
	read_file_path = base_floder_path + "/extra_info.txt"
	with open(read_file_path) as f:
		_ = f.readline()
		attrs_value = f.readline().replace("\n", "").split(" ")
		segment_amount = int(attrs_value[0])
	read_num = 0
	database = []
	for segment_number in range(segment_amount):
		if is_debug is True:
			print("now read file: " + str(segment_number))
		if numRead != -1 and read_num >= numRead:
			break
		eds =ECG_operate()
		eds.global_id = segment_number
		eds.read_ecg_segment(rdf, database_name)
		database.append(eds)
		read_num += 1
	if is_debug is True:
		print("length of database: %s" % len(database))
	return database

def interp_cubic_spline(rri, fs):
	time_rri = create_time_info(rri)
	time_rri_interp = create_interp_time(rri, fs)
	tck_rri = interpolate.splrep(time_rri, rri, s=0)
	rri_interp = interpolate.splev(time_rri_interp, tck_rri, der=0)
	return rri_interp


def interp_cubic_spline_qrs(qrs_index, qrs_amp, fs):
	time_qrs = qrs_index / float(ECGFrequency)
	time_qrs = time_qrs - time_qrs[0]
	time_qrs_interp = np.arange(0, 60, 1 / float(fs))
	tck = interpolate.splrep(time_qrs, qrs_amp, s=0)
	qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)
	return qrs_interp


def smooth(a, WSZ):
	out0 = np.convolve(a, np.ones(WSZ, dtype=float), 'valid') / WSZ
	r = np.arange(1, WSZ - 1, 2)
	start = np.cumsum(a[:WSZ - 1])[::2] / r
	stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
	return np.concatenate((start, out0, stop))


def rricheck(ecg_data, rr_intervals):
	noise_flag = rr_intervals > 180
	noise_flag1 = rr_intervals < 30
	if len(rr_intervals) < 40 \
			or np.sum(noise_flag) > 0 \
			or np.sum(noise_flag1) > 0 \
			or len(ecg_data) != 6000:
		return False
	else:
		return True


def compute_r_peak_amplitude(ecg_data, rwave):

	wave_amp = []
	for peak_ind in rwave.tolist():
		interval = 25
		if peak_ind - interval < 0:
			start = 0
		else:
			start = peak_ind - interval
		
		if peak_ind + interval > len(ecg_data):
			end = len(ecg_data)
		else:
			end = peak_ind + interval
		amp = np.max(ecg_data[start:end])
		wave_amp.append(amp)
	return np.array(wave_amp)


def denoising(ecg_segment):
	denoising_wd_level = 6
	denoising_wd_wavelet = "db6"
	coffes_set = []
	cA_signal = np.reshape(ecg_segment, len(ecg_segment))
	for index_dec in range(denoising_wd_level):
		cA, cD = pywt.dwt(cA_signal, denoising_wd_wavelet)
		coffes_set.append(cD)
		cA_signal = cA
	coffes_set.append(cA_signal)
	coffes_set[denoising_wd_level] = np.zeros(len(coffes_set[denoising_wd_level]))
	cA_signal = coffes_set[denoising_wd_level]
	for index_dec in range(denoising_wd_level):
		cD_signal = coffes_set[denoising_wd_level - 1 - index_dec]
		if len(cD_signal) != len(cA_signal):
			cA_signal = np.delete(cA_signal, len(cA_signal) - 1, axis=0)
		cA_signal = pywt.idwt(cA_signal, cD_signal, denoising_wd_wavelet)
	cA_signal = np.reshape(cA_signal, (len(ecg_segment), 1))
	return cA_signal


def pre_proc(dataset, database_name, is_debug=False):
	clear_id_set, noise_id_set = [], []
	for segment in dataset:
		if is_debug:
			print("now process %s	id=%s." % (segment.database_name, str(segment.global_id)))
		segment.denoised_ecg_data = denoising(segment.raw_ecg_data)
		segment.write_ecg_segment(rdf=1)
		name = segment.base_file_path + 'denoised_ecg_data.mat'
		sio.savemat(name, {'denoised_ecg_data': segment.denoised_ecg_data})
		eng.getfeaturedata(segment.base_file_path)
		if os.path.exists(segment.base_file_path + "/Rwave.mat"):
			RwaveMat = sio.loadmat(segment.base_file_path + "/Rwave.mat")
			Rwave = np.transpose(RwaveMat['Rwave'])
			Rwave = np.reshape(Rwave, len(Rwave))
			RR_intervals = np.diff(Rwave)
			np.save(segment.base_file_path + "/RRI.npy", RR_intervals)
			rri_flag = rricheck(segment.denoised_ecg_data, RR_intervals)
			if rri_flag:
				clear_id_set.append(segment.global_id)
			else:
				noise_id_set.append(segment.global_id)
				continue
			RAMP = compute_r_peak_amplitude(segment.denoised_ecg_data, Rwave)
			RRI = smooth(RR_intervals, 3)
			RAMP = smooth(RAMP, 3)
			RRI = RRI / ECGFrequency * 1000.0
			RRI = interp_cubic_spline(RRI, fs=4)
			RAMP = interp_cubic_spline_qrs(Rwave, RAMP, fs=4)
			np.save(segment.base_file_path + "/RRI.npy", RRI)
			np.save(segment.base_file_path + "/RAMP.npy", RAMP)
			# EDRMat = sio.loadmat(segment.base_file_path + "/EDR.mat")
			# EDR = np.transpose(EDRMat['EDR'])
			# EDR = np.reshape(EDR, len(EDR))
			# EDR = decimate(EDR, 25)
			# np.save(segment.base_file_path + "/EDR.npy", EDR)
		else:
			noise_id_set.append(segment.global_id)
	print(len(noise_id_set))
	print(len(clear_id_set))
	np.save(database_name[0] + "_" + database_name[1] + "_clear_id.npy", np.array(clear_id_set))
	np.save(database_name[0] + "_" + database_name[1] + "_noise_id.npy", np.array(noise_id_set))
	

if __name__ == '__main__':
	train_set = load_data(["apnea-ecg", "train"], rdf=0,is_debug=True)
	pre_proc(train_set, ["apnea-ecg", "train"], is_debug=True)
	test_set = load_data(["apnea-ecg", "test"], rdf=0, is_debug=True)
	pre_proc(test_set, ["apnea-ecg", "test"], is_debug=True)
	



