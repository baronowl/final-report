
import wfdb
import os
import numpy as np
from datapreprocessing.operate import ECG_operate

#Has used some code and algorithms from :https://github.com/zzklove3344/ApneaECGAnalysis/tree/master/preprocessOfApneaECG

Apnea_ECG_data_path = "D:/#D/Apean-ECG-database/"
Base_path = "D:/#D/ecg-separate/"
Save_train_path = "D:/#D/ecg-separate/train"
Save_test_path = "D:/#D/ecg-separate/test"

train_data = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10","a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20","b01", "b02", "b03", "b04", "b05","c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"]

test_data = ["x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10","x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30","x31", "x32", "x33", "x34", "x35"]

Total = [523, 469, 465, 482, 505,450, 509, 517, 508, 510,457, 527, 506, 490, 498, 515, 400, 459, 487, 513,510, 482, 527, 429, 510, 520, 498, 495, 470, 511,557, 538, 473, 475, 483]


def ECG_TXT(database_name):
	data_annotations_set = []
	file_name_set = None
	no_apn = None
	data_set = []
	global_counter = 0

	base_floder_path = None
	if database_name[0] == "apnea-ecg":
		root_file_path = Apnea_ECG_data_path
		if database_name[1] == "train":
			file_name_set = train_data
			base_floder_path = Save_train_path
			no_apn = False
		elif database_name[1] == "test":
			file_name_set = test_data
			base_floder_path = Save_test_path
			no_apn = True

	test_label_set = []
	if no_apn is True:
		# read event-2.txt file from PhysioNet
		test_annotation_path = root_file_path + "event-2.txt"
		with open(test_annotation_path) as f:
			lines = f.readlines()
			for line in lines:
				line = line.replace("\n", "")
				for index_str in range(len(line)):
					if line[index_str] == "A" or line[index_str] == "N":
						test_label_set.append(line[index_str])

	file_count = 0
	test_label_index = 0
	for name in file_name_set:
		file_path = root_file_path + name
		ecg_data = wfdb.rdrecord(file_path)
		if no_apn is False:
			annotation = wfdb.rdann(file_path, "apn")
			annotation_range_list = annotation.sample
			annotation_list = annotation.symbol
		else:
			annotation_range_list = []
			annotation_list = []
			for index_label in range(Total[file_count]):
				annotation_range_list.append(np.array(index_label * 6000))
				annotation_list.append(test_label_set[test_label_index])
				test_label_index += 1
			file_count += 1
			annotation_range_list = np.array(annotation_range_list)

		data_annotations_set.append([ecg_data, annotation_range_list, annotation_list, name])

	for data_annotation in data_annotations_set:
		segment_amount = len(data_annotation[2])
		for index_segment in range(segment_amount):
			eds = ECG_operate()
			eds.database_name = database_name
			eds.samplefrom = data_annotation[1][index_segment]
			if (data_annotation[1][index_segment] + 6000) > len(data_annotation[0].p_signal):
				eds.sampleto = len(data_annotation[0].p_signal)
			else:
				eds.sampleto = data_annotation[1][index_segment] + 6000
			eds.raw_ecg_data = data_annotation[0].p_signal[eds.samplefrom:eds.sampleto]
			eds.label = data_annotation[2][index_segment]
			eds.filename = data_annotation[3]
			eds.local_id = index_segment
			eds.global_id = global_counter
			eds.base_file_path = "D:/#D/ecg-separate/" + database_name[1] + "/" + str(eds.global_id) + "/"
			eds.write_ecg_segment(rdf=0)
			global_counter += 1
			data_set.append(eds)
	if not os.path.exists(base_floder_path):
		os.makedirs(base_floder_path)
	with open(base_floder_path + "/extra_info.txt", "w") as f:
		f.write("Number of ECG segments\n")
		f.write(str(global_counter))


if __name__ == '__main__':
	print("Start processing...")
	ECG_TXT(["apnea-ecg", "train"])
	ECG_TXT(["apnea-ecg", "test"])
	print("Process finish...")
