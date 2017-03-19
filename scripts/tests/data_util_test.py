from data_util import DataManager
import numpy as np
DataInstance = DataManager("../../data/train", "../../data/test", 128)
batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
print (np.shape(batch_input))
