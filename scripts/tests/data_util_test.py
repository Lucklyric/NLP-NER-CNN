import sys
import numpy as np

sys.path.append("../")
import data_util_v3
from data_util import DataManager
from data_util_v3 import DataManager as DataManager_v3
# DataInstance = DataManager("../../data/train", "../../data/test", 128)
# batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# print (np.shape(batch_input))
data_util_v3.save_to_disk("../../data/train", "../../data/test")
DataInstance = DataManager_v3("../../data/train_np_v3.npy", "../../data/test_np_v3.npy", 128)
batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
print (np.shape(batch_input))
print (np.shape(batch_output))
sample_input, sample_output = DataInstance.get_one_sample(0)
print (sample_output)
# DataInstance = DataManager("../../data/train_np.npy", "../../data/test_np.npy", 64)
# print (np.shape(DataInstance._train_data))
