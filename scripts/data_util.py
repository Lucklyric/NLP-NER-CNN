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
        sentence_data = np.zeros((70 + 12, 500), dtype=np.float32)
        col_idx = 0
        for word_idx in range(len(sentence)):
            word = sentence[word_idx]
            target_symbol_index = 70  # 0 PASS
            if ("company" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 71
            elif ("facility" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 72
            elif ("geo-loc" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 73
            elif ("movie" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 74
            elif ("musicartist" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 75
            elif ("other" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 76
            elif ("person" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 77
            elif ("product" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 78
            elif ("sportsteam" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 79
            elif ("tvshow" in target_sentences[sentence_idx][word_idx]) is True:
                target_symbol_index = 80
            for char in word.upper():  # upper the
                char_dec = ord(char)
                row_idx = 68  # represent other unkonw symbols
                if 96 >= char_dec >= 33:
                    row_idx = char_dec - 33
                elif 126 >= char_dec >= 123:
                    row_idx = char_dec - 33 - 26
                sentence_data[row_idx, col_idx] = 1
                sentence_data[target_symbol_index, col_idx] = 1
                col_idx += 1
            sentence_data[69, col_idx] = 1
            sentence_data[81, col_idx] = 1
            col_idx += 1
        data.append(sentence_data)
    return np.array(data)


class DataManager(object):
    def __init__(self, train_data, evl_data, batch_size):
        self._train_data = train_data
        self._evl_data = evl_data
        self._train_data = parse_raw_data(self._train_data)
        self._evl_data = parse_raw_data(self._evl_data)
        self._batch_size = batch_size
        self._batch_index = 0

    def get_batch(self):
        epoch_end = False
        self._batch_index += self._batch_size
        if self._batch_index > np.shape(self._train_data)[0]:
            epoch_end = True
            np.random.shuffle(self._train_data)  # shuffle the data
            self._batch_index = self._batch_size
        batch_data = self._train_data[self._batch_index - self._batch_size:self._batch_index]
        batch_input = batch_data[:, 0:70, :]
        batch_output = batch_data[:, 70:, :]
        return batch_input, batch_output, epoch_end
