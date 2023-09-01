from keras import losses, optimizers, Input, models, layers
from transformers import TFBertModel
from transformers import BertTokenizerFast
import tensorflow as tf


class CustomModel():
    def __init__(self, model:models.Model, loss=None, opt=None) -> None:
        self.model = model
        self.loss = loss
        self.opt = opt
        
    def build(self, summary=False):
        pass
    def predict(input=None):
        pass
    def fit(self, train_dataset, dev_dataset=None, epochs=1, callbacks=None):
        pass
    def getConfig(self):
        pass
    
class NerBERT(CustomModel):
    def __init__(self, 
                 tags_map:tf.keras.preprocessing.text.Tokenizer,
                 name="NerBERT",
                 max_len=128,
                 rate_drop=0.5,
                 decode='utf-8',
                 path_or_name_model='bert-base-uncased',
                 loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
                 opt=optimizers.Adam()):
        super().__init__(model=None, loss=loss, opt=opt)
        
        self.name = name
        self.tags_map = tags_map
        self.path_or_name_model = path_or_name_model
        self.max_len = max_len
        self.rate_drop = rate_drop
        self.num_tags = len(tags_map.index_word) + 1
        self.decode = decode
        self.tokenizer = BertTokenizerFast.from_pretrained(self.path_or_name_model)
        
        
    def encodeSeq(self, input):
        encoded = self.tokenizer.encode_plus(input,
                                             add_special_tokens=True,
                                             max_length=self.max_len,
                                             is_split_into_words=True,
                                             return_attention_mask=True,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='np')
        return encoded['input_ids'], encoded['attention_mask']
    
    def build(self, summary=False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_ids = Input(shape=(self.max_len, ), dtype='int32')
            attention_masks = Input(shape=(self.max_len, ), dtype='int32')
            
            bert_model = TFBertModel.from_pretrained(self.path_or_name_model)
            bert_output = bert_model(input_ids, attention_mask=attention_masks, return_dict=True)
            
            embedding = layers.Dropout(self.rate_drop)(bert_output["last_hidden_state"])
            output = layers.Dense(self.num_tags, activation = 'softmax')(embedding)
            
            model = models.Model(inputs=[input_ids, attention_masks], outputs=[output], name=self.name)
            model.compile(optimizer=self.opt, loss=self.loss, metrics=["accuracy"])
        if summary:
            model.summary()
        self.model = model
        return self
    
    def fit(self, train_dataset, dev_dataset=None, epochs=1, batch_size=1, validation_batch_size=None, callbacks=None):
        X,y = train_dataset
        self.model.fit(X,y,
                       batch_size=batch_size,
                       validation_data=dev_dataset,
                       validation_batch_size=validation_batch_size, 
                       epochs=epochs,
                       callbacks=callbacks)
        return self
    
    def decoderTags(self, output_tf):
        output_tf = tf.math.argmax(output_tf, axis=-1)
        output = tf.squeeze(output_tf).numpy()
        output = self.tags_map.sequences_to_texts([output])
        return output
    
    def formatOutput(self, output_tf, input_size=0):
        output = self.decoderTags(output_tf)
        output = str(output[0]).split()
        output = output[:input_size]
        return output

    
    def predict(self, input=None):
        input_size = len(input.split())
        input_ids, attention_mask = self.encodeSeq(input=input)
        output_tf = self.model.predict_on_batch([input_ids, attention_mask])
        output = self.formatOutput(output_tf, input_size)
        return list(zip(input.split(), output))