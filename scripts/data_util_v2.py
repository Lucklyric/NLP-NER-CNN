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

    data = []
    for sentence_idx in range(len(input_sentences)):
        sentence = input_sentences[sentence_idx]
        sentence_data = np.zeros((71 + 1, 400), dtype=np.float32)
        col_idx = 0
        for word_idx in range(len(sentence)):
            word = sentence[word_idx]
            target_symbol = 0  # 0 PASS
            if ("company" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 1
            elif ("facility" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 2
            elif ("geo-loc" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 3
            elif ("movie" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 4
            elif ("musicartist" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 5
            elif ("other" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 6
            elif ("person" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 7
            elif ("product" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 8
            elif ("sportsteam" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 9
            elif ("tvshow" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol = 10
            for char in word.upper():  # upper the
                char_dec = ord(char)
                row_idx = 68  # represent other unkonw symbols
                if 96 >= char_dec >= 33:
                    row_idx = char_dec - 33
                elif 126 >= char_dec >= 123:
                    row_idx = char_dec - 33 - 26
                sentence_data[0:row_idx, col_idx] = 1
                sentence_data[71, col_idx] = target_symbol
                col_idx += 1
            sentence_data[69, col_idx] = 1
            sentence_data[71, col_idx] = 11
            col_idx += 1
        sentence_data[70, col_idx:] = 1
        sentence_data[71, col_idx:] = 12
        data.append(sentence_data)
    return np.array(data)


def save_to_disk(train_data, evl_data):
    np.save(train_data + "_np_v2", parse_raw_data(train_data))
    np.save(evl_data + "_np_v2", parse_raw_data(evl_data))


class DataManager(object):
    def __init__(self, train_data, evl_data, batch_size):
        print ("Start loading data ...")
        self._train_data = train_data
        self._evl_data = evl_data
        self._train_data = np.load(self._train_data)
        self._evl_data = np.load(self._evl_data)
        self._batch_size = batch_size
        self._batch_index = 0
        print ("Data loaded !")

    def get_one_sample(self, index=0, source="test"):
        if source != "test":
            return self._train_data[index, 0:71, :], self._train_data[index, 71:, :]
        else:
            return self._evl_data[index, 0:71, :], self._evl_data[index, 71, :]

    def get_batch(self):
        epoch_end = False
        self._batch_index += self._batch_size
        if self._batch_index > np.shape(self._train_data)[0]:
            epoch_end = True
            np.random.shuffle(self._train_data)  # shuffle the data
            self._batch_index = self._batch_size
        batch_data = self._train_data[self._batch_index - self._batch_size:self._batch_index]
        batch_input = batch_data[:, 0:71, :]
        batch_output = batch_data[:, 71, :]
        return batch_input, batch_output, epoch_end
