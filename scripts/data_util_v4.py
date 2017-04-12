import numpy as np


def parse_raw_data(path):
    input_sentences = []
    target_sentences = []
    with open(path) as f:
        in_sentence = []
        target_sentence = []
        for line in f:
            if line != "\n":
                in_target = line.split('\t')
                in_sentence.append(in_target[0])
                target_sentence.append(in_target[1].strip())
            else:
                input_sentences.append(in_sentence)
                target_sentences.append(target_sentence)
                in_sentence = []
                target_sentence = []

    input_data = []
    output_data = []
    for sentence_idx in range(len(input_sentences)):
        sentence = input_sentences[sentence_idx]
        sentence_in_data = np.zeros([50, 70, 20], dtype=np.float32)
        sentence_out_data = np.zeros([12, 50], dtype=np.float32)
        word_idx = 0
        for word in sentence:
            if word_idx >= 50:
                break
            # handle target output
            target_symbol_index = 0  # 0 PASS
            if ("company" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 1
            elif ("facility" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 2
            elif ("geo-loc" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 3
            elif ("movie" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 4
            elif ("musicartist" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 5
            elif ("other" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 6
            elif ("person" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 7
            elif ("product" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 8
            elif ("sportsteam" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 9
            elif ("tvshow" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 10

            sentence_out_data[target_symbol_index, word_idx] = 1

            # handle input word
            col_idx = 0
            for char in word.upper():  # upper the
                if col_idx >= 20:
                    break
                char_dec = ord(char)
                row_idx = 68  # represent other unkonw symbols
                if 96 >= char_dec >= 33:
                    row_idx = char_dec - 33
                elif 126 >= char_dec >= 123:
                    row_idx = char_dec - 33 - 26
                sentence_in_data[word_idx, 0:row_idx, col_idx] = 1
                col_idx += 1

            word_idx += 1
        sentence_in_data[word_idx:, 69, :] = 1  # PAD
        sentence_out_data[11, word_idx:] = 1  # PAD
        input_data.append(sentence_in_data)
        output_data.append(sentence_out_data)
    return np.array(input_data), np.array(output_data)


def save_to_disk(train_data, evl_data):
    train_in, train_out = parse_raw_data(train_data)
    np.save(train_data + "_in_np_v4", train_in)
    np.save(train_data + "_out_np_v4", train_out)

    evl_in, evl_out = parse_raw_data(evl_data)
    np.save(evl_data + "_in_np_v4", evl_in)
    np.save(evl_data + "_out_np_v4", evl_out)


def final_evaluate(test_output, target_output):
    total_token = 0
    class_tokens_total = np.zeros(11, dtype=np.int8)
    class_tokens_TP = np.zeros(11, dtype=np.int8)
    class_tokens_TN = np.zeros(11, dtype=np.int8)
    class_tokens_FP = np.zeros(11, dtype=np.int8)
    class_tokens_FN = np.zeros(11, dtype=np.int8)
    for s_index in range(len(test_output)):
        sentence = test_output[s_index]
        sentence_target = target_output[s_index]
        for w_index in range(len(sentence)):
            output_label = np.argmax(sentence[:, w_index])
            target_label = np.argmax(sentence_target[:, w_index])
            if target_label == 11:
                break  # skip left if reach PAD
            total_token += 1  # add total token
            class_tokens_total[target_label] += 1
            if target_label == output_label:
                class_tokens_TP[output_label] += 1
                class_tokens_TN[:] += 1
                class_tokens_TN[output_label] += 1
            if target_label != output_label:
                class_tokens_FN[target_label] += 1
                if output_label != 12:
                    class_tokens_FP[output_label] += 1

    # Output Table
    print ("--------------------------------------------------")
    for i in range(11):
        print ("%d  TP: %d, TN: %d, FP: %d, FN: %d, Total: %d" % (i,
                                                                  class_tokens_TP[i],
                                                                  class_tokens_TN[i],
                                                                  class_tokens_FP[i],
                                                                  class_tokens_FN[i],
                                                                  class_tokens_total[i]))


class DataManager(object):
    def __init__(self, train_data_in, train_data_out, evl_data_in, evl_data_out, batch_size):
        print ("Start loading data ...")
        self._train_data_in = np.load(train_data_in)
        self._train_data_out = np.load(train_data_out)
        self._evl_data_in = np.load(evl_data_in)
        self._evl_data_out = np.load(evl_data_out)
        self._batch_size = batch_size
        self._batch_index = 0
        print ("Data loaded !")

    def get_one_sample(self, index=0, source="test"):
        if source != "test":
            return self._train_data_in[index, :, :, :], self._train_data_out[index, :, :]
        else:
            return self._evl_data_in[index, :, :, :], self._evl_data_out[index, :, :]

    def get_eval_data(self):
        return self._evl_data_in, self._evl_data_out

    def get_batch(self):
        epoch_end = False
        self._batch_index += self._batch_size
        if self._batch_index > len(self._train_data_in):
            epoch_end = True
            randomize = np.arange(len(self._train_data_in))
            np.random.shuffle(randomize)
            self._train_data_in = self._train_data_in[randomize]
            self._train_data_out = self._train_data_out[randomize]
            self._batch_index = self._batch_size
        batch_input = self._train_data_in[self._batch_index - self._batch_size:self._batch_index]
        batch_output = self._train_data_out[self._batch_index - self._batch_size:self._batch_index]
        return batch_input, batch_output, epoch_end
