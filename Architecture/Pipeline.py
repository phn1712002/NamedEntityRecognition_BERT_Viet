import tensorflow as tf
import numpy as np
from Architecture.Model import NerBERT


class PipelineBERT(NerBERT):
    def __init__(self, tags_map:tf.keras.preprocessing.text.Tokenizer, config_model=None):
        super().__init__(tags_map=tags_map, **config_model)
        
    def tokenize(self, data):
        input_ids_list = list()
        attention_mask_list = list()
        for i in range(len(data)):
            input_ids, attention_mask = self.encodeSeq(data[i])
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        return np.vstack(input_ids_list), np.vstack(attention_mask_list)
    
    def formatSeq(self, data):
        return [list(string.split()) for string in data]
    
    def encoderTag(self, data):
        data = self.tags_map.texts_to_sequences(data)
        data = tf.keras.utils.pad_sequences(data, padding='post', maxlen=self.max_len)
        return data
    
    def __call__(self, dataset):
        if not dataset is None:
            seq, tag = dataset
            return self.tokenize(self.formatSeq(seq)), self.encoderTag(tag)
        else:
            return None