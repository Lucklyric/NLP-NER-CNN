import sys
import numpy as np

sys.path.append("../")
# from data_util import DataManager
# from data_util_v3 import DataManager as DataManager_v3
# from data_util_v2 import DataManager as DataManager_v2
# import data_util_v2
# import data_util_v3
# import data_util_v4
# # DataInstance = DataManager("../../data/train", "../../data/test", 128)
# # batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# # print (np.shape(batch_input))
# data_util_v2.save_to_disk("../../data/train", "../../data/test")
# DataInstance = DataManager_v2("../../data/train_np_v2.npy", "../../data/test_np_v2.npy", 128)
# batch_input, batch_output, is_epoch_increase = DataInstance.get_batch()
# print (np.shape(batch_input))
# print (np.shape(batch_output))
# sample_input, sample_output = DataInstance.get_one_sample(0)
# print (sample_input)
# print (sample_output)
# DataInstance = DataManager("../../data/train_np.npy", "../../data/test_np.npy", 64)
# print (np.shape(DataInstance._train_data))

# V4 Test
import data_util_v5

# Save to disk
# data_util_v5.save_to_disk("../../data/train","../../data/test")
DataInstance = data_util_v5.DataManager("../../data/train_in_np_v5.npy", "../../data/train_out_np_v5.npy",
                                        "../../data/test_in_np_v5.npy", "../../data/test_out_np_v5.npy", 2)
sample_input, sample_output = DataInstance.get_one_sample(0)
print sample_output
print np.shape(sample_input)
print np.shape(sample_output)
