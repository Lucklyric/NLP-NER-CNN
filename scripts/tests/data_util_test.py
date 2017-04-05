import sys
import numpy as np

sys.path.append("../")
import data_util

# DataInstance = DataManager("../../data/train", "../../data/test", 128)
# batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# print (np.shape(batch_input))
data_util.save_to_disk("../../data/train", "../../data/test")
