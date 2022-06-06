
import numpy as np
from datapreprocessing.operate import ECG_operate
import os


class ECGSegment(ECG_operate):

	def __init__(self):
		super(ECGSegment, self).__init__()
		self.RR_intervals = []
		self.R_peaks_amplitude = []

	def read_rri_ramp(self):
		self.RR_intervals = np.load(self.base_file_path + "/RRI.npy")
		self.R_peaks_amplitude = np.load(self.base_file_path + "/RAMP.npy")

def cearate_featuredata(database_name):
	if database_name == ["apnea-ecg", "train"] or database_name == ["apnea-ecg", "test"]:
		clear_id_set = np.load(database_name[0] + "_" + database_name[1] + "_clear_id.npy")
	else:
		raise Exception("Error file.")
	dataset = []
	RRI_set = []
	RAMP_set = []
	label_set = []
	for id in clear_id_set:
		eds = ECGSegment()
		eds.global_id = id
		eds.read_ecg_segment(1, database_name)
		eds.read_rri_ramp()
		label_set.append(eds.label)
		RRI_set.append(eds.RR_intervals)
		RAMP_set.append(eds.R_peaks_amplitude)
		dataset.append(eds)
	print(RRI_set)
	mean = np.mean(RRI_set, axis=1)
	mean = np.reshape(mean, (mean.shape[0], 1))
	rri_set = RRI_set - mean
	rri_set = np.reshape(rri_set, (rri_set.shape[0], rri_set.shape[1], 1))
	mean = np.mean(RAMP_set, axis=1)
	mean = np.reshape(mean, (mean.shape[0], 1))
	ramp_set = RAMP_set - mean
	ramp_set = np.reshape(ramp_set, (ramp_set.shape[0], ramp_set.shape[1], 1))
	ramp_set = ramp_set * 100
	rri_amp_set = np.concatenate([rri_set, ramp_set], axis=2)
	
	if not os.path.exists("test/"):
		os.makedirs("test/")
	np.save("test/" + database_name[1] + "_rri.npy", np.array(rri_set))
	np.save("test/" + database_name[1] + "_ramp.npy", np.array(ramp_set))
	np.save("test/" + database_name[1] + "_label.npy", np.array(label_set))
	np.save("test/" + database_name[1] + "_rri_ramp.npy", np.array(rri_amp_set))


if __name__ == '__main__':
	print("Start producing...")
	cearate_featuredata(["apnea-ecg", "train"])
	cearate_featuredata(["apnea-ecg", "test"])
	
		
