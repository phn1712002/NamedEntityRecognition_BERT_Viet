import argparse, warnings, wandb
import tensorflow as tf
from Tools.Json import loadJson
from Tools.Callbacks import CreateCallbacks
from Tools.Weights import loadNearest, loadWeights
from Tools.TFLite import convertModelKerasToTflite
from Architecture.Model import NerBERT
from Architecture.Pipeline import PipelineBERT
from Dataset.Createdataset import DatasetNerBERT
from Optimizers.OptimizersNerBERT import CustomOptimizers

# Environment variable
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_SAVE = './Checkpoint/save/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TENSORBOARD = './Checkpoint/tensorboard/'
PATH_TFLITE = './Checkpoint/export/'

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=bool, default=False, help='Pretrain model BERT in logs training in dataset')
parser.add_argument('--path_file_pretrain', type=str, default='', help='Path file pretrain model')
parser.add_argument('--export_tflite', type=bool, default=False, help='Export to tflite')
args = parser.parse_args()


# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_model', 'config_opt', 'config_other', 'config_train']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_model = config['config_model']
        config_opt = config['config_opt']
        config_other = config['config_other']
        config_train = config['config_train']
    else:
        raise RuntimeError('Error config')

# Turn off warning
if not config_other['warning']:
    warnings.filterwarnings('ignore')
    
# Load dataset
train_dataset_raw, dev_dataset_raw, test_dataset_raw = DatasetNerBERT(path=PATH_DATASET, encoding=config_model['decode'])()

# Init tags
tags_map = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, split=' ', oov_token='UNK')
tags_map.fit_on_texts(train_dataset_raw[1])

# Create pipeline 
pipeline = PipelineBERT(tags_map=tags_map, config_model=config_model)
train_dataset = PipelineBERT(tags_map=tags_map, config_model=config_model)(dataset=train_dataset_raw)
dev_dataset = PipelineBERT(tags_map=tags_map, config_model=config_model)(dataset=dev_dataset_raw)


# Create optimizers
opt_biLSTM = CustomOptimizers(**config_opt)()

# Callbacks
callbacks_NER = CreateCallbacks(PATH_TENSORBOARD=PATH_TENSORBOARD, 
                                PATH_LOGS=PATH_LOGS, 
                                config=config, 
                                train_dataset=train_dataset, 
                                dev_dataset=dev_dataset, 
                                pipeline=pipeline)

# Create model
ner = NerBERT(path_or_name_model=PATH_SAVE, tags_map=tags_map, **config_model).build(config_other['summary'])

# Pretrain
if args.pretrain_config:
    if args.path_file_pretrain == '':
        ner = loadNearest(class_model=ner, path_folder_logs=PATH_LOGS)
    else: 
        ner = loadWeights(class_model=ner, path=args.path_file_pretrain)

# Train model
ner.fit(train_dataset=train_dataset,
        batch_size=config_train['batch_size_train'],
        dev_dataset=dev_dataset,
        validation_batch_size=config_train['batch_size_dev'],
        epochs=config_train['epochs'], 
        callbacks=callbacks_NER)
"""
# Export to tflite
if args.export_tflite:
    convertModelKerasToTflite(class_model=ner, path=PATH_TFLITE)
"""
# Off Wandb
wandb.finish()