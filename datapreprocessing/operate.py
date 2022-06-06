
import os
import numpy as np

#Has used some code and algorithms from :https://github.com/zzklove3344/ApneaECGAnalysis/tree/master/preprocessOfApneaECG

Base_path = "D:/#D/ecg-separate/"
# Base_path = "D:/#D/ecg-p1/"
Save_train_path = "D:/#D/ecg-separate/train"
Save_test_path = "D:/#D/ecg-separate/test"



class ECG_operate:
	
	def __init__(self):
		self.raw_ecg_data = None
		self.denoised_ecg_data = None
		self.label = None
		self.database_name = None
		self.filename = None
		self.local_id = None
		self.global_id = None
		self.samplefrom = None
		self.sampleto = None
		self.base_file_path = None
		
	def write_ecg_segment(self, rdf):
		if not os.path.exists(self.base_file_path):
			os.makedirs(self.base_file_path)
		if rdf == 0:
			filename = "raw_ecg_segment_data.txt"
			ecg_data = self.raw_ecg_data
		elif rdf == 1:
			filename = "denosing_ecg_segment_data.txt"
			ecg_data = self.denoised_ecg_data
		else:
			raise Exception("Error rdf value.")
		
		attr_name = "database_name file_name local_id samplefrom sampleto global_id label\n"
		if self.label == 'A':
			self.label = 1
		elif self.label == 'N':
			self.label = 0
		
		with open(self.base_file_path + filename, "w") as f:
			r"""attributes name """
			f.write(attr_name)
			
			r"""attributes value"""
			f.write(
				self.database_name[0] + " " + self.database_name[1] + " "
				+ self.filename + " " + str(self.local_id) + " "
				+ str(self.global_id) + " " + str(self.samplefrom) + " "
				+ str(self.sampleto) + " " + str(self.label) + "\n")
			
			r"""data"""
			for value in ecg_data:
				f.write(str(value[0]) + "\n")
	
	def read_ecg_segment(self, rdf, database_name_or_path):
		if rdf == 0:
			filename = "raw_ecg_segment_data.txt"
		elif rdf == 1:
			filename = "denosing_ecg_segment_data.txt"
		else:
			raise Exception("Error rdf value.")
		if database_name_or_path == ["apnea-ecg", "train"]:
			file_path = Base_path + "train/" + str(self.global_id) + "/" + filename
		elif database_name_or_path == ["apnea-ecg", "test"]:
			file_path = Base_path + "test/" + str(self.global_id) + "/" + filename
		else:
			file_path = database_name_or_path
		
		with open(file_path) as f:
			_ = f.readline()
			attrs_value = f.readline().replace("\n", "").split(" ")
			self.database_name = [attrs_value[0], attrs_value[1]]
			self.filename = attrs_value[2]
			self.local_id = int(attrs_value[3])
			self.global_id = int(attrs_value[4])
			self.samplefrom = int(attrs_value[5])
			self.sampleto = int(attrs_value[6])
			self.label = int(attrs_value[7])
			self.base_file_path = Base_path + self.database_name[1] + "/" + str(self.global_id) + "/"

			ecg_data = []
			data_value = f.readline().replace("\n", "")
			while data_value != "":
				ecg_data.append(float(data_value))
				data_value = f.readline().replace("\n", "")
			if rdf == 0:
				self.raw_ecg_data = ecg_data
			elif rdf == 1:
				self.denoised_ecg_data = ecg_data
	
	def read_edr(self, flag):
		edr = []
		if self.filename.find('x') >= 0:
			if flag == 0:
				file_path = Base_path + "test/" + str(self.global_id) + "/edr.txt"
			elif flag == 1:
				file_path = Base_path + "test/" + str(self.global_id) + "/downsampling_EDR.txt"
			else:
				file_path = ""
				print("edr file path error....")
		else:
			if flag == 0:
				file_path = Base_path + "train/" + str(self.global_id) + "/edr.txt"
			elif flag == 1:
				file_path = Base_path + "train/" + str(self.global_id) + "/downsampling_EDR.txt"
			else:
				file_path = ""
				print("edr file path error....")
		
		with open(file_path) as f:
			data_value = f.readline().replace("\n", "")
			while data_value != "":
				edr.append(float(data_value))
				data_value = f.readline().replace("\n", "")
			edr = np.array(edr)
		return edr


