from data_util import DataManager
import data_util
import numpy as np

# DataInstance = DataManager("../../data/train", "../../data/test", 128)
# batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# print (np.shape(batch_input))
data_util.save_to_disk("../../data/train", "../../data/test")
