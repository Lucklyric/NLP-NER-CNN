import sys
import numpy as np

sys.path.append("../")
import data_util
from data_util import DataManager

# DataInstance = DataManager("../../data/train", "../../data/test", 128)
# batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# print (np.shape(batch_input))
# data_util.save_to_disk("../../data/train", "../../data/test")
DataInstance = DataManager("../../data/train_np.npy", "../../data/test_np.npy", 64)
sample_input, sample_output = DataInstance.get_one_sample(0)
print (np.shape(DataInstance._train_data))
