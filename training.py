from dataset import make_dataset, custom_standardization, reduce_dataset_dim, valid_test_split
#from settings import *
from config import get_config
from custom_schedule import custom_schedule
from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from utility import save_tokenizer
import json
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

config = get_config()

# Load dataset
with open(config.train_data_json_path) as json_file:
    train_data = json.load(json_file)
with open(config.valid_data_json_path) as json_file:
    valid_data = json.load(json_file)
with open(config.test_data_json_path) as json_file:
    test_data = json.load(json_file)
with open(config.text_data_json_path) as json_file:
    text_data = json.load(json_file)

# For reduce number of images in the dataset
if config.REDUCE_DATASET:
    train_data, valid_data = reduce_dataset_dim(config, train_data, valid_data)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))
print("Number of test samples: ", len(test_data))

# Define tokenizer of Text Dataset
tokenizer = keras.layers.TextVectorization(
    max_tokens=config.MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=config.SEQ_LENGTH,
    standardize=custom_standardization,
)

# Adapt tokenizer to Text Dataset
tokenizer.adapt(text_data)

# Define vocabulary size of Dataset
VOCAB_SIZE = len(tokenizer.get_vocabulary())
#print(VOCAB_SIZE)

# 20k images for validation set and 13432 images for test set
#valid_data, test_data  = valid_test_split(valid_data)

# Setting batch dataset
train_dataset = make_dataset(config, list(train_data.keys()), list(train_data.values()), data_aug=config.TRAIN_SET_AUG, tokenizer=tokenizer)
valid_dataset = make_dataset(config, list(valid_data.keys()), list(valid_data.values()), data_aug=config.VALID_SET_AUG, tokenizer=tokenizer)
if config.TEST_SET:
    test_dataset = make_dataset(config, list(test_data.keys()), list(test_data.values()), data_aug=False, tokenizer=tokenizer)

# Define Model
cnn_model = get_cnn_model(config.IMAGE_SIZE, config.cnnmodel)

encoder = TransformerEncoderBlock(
    embed_dim=config.EMBED_DIM, dense_dim=config.FF_DIM, num_heads=config.NUM_HEADS
)
decoder = TransformerDecoderBlock(
    embed_dim=config.EMBED_DIM, ff_dim=config.FF_DIM, num_heads=config.NUM_HEADS, vocab_size=VOCAB_SIZE, seq_length=config.SEQ_LENGTH
)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, num_captions_per_image=config.num_captions_per_image
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'{config.SAVE_DIR}/best_model.keras',  # Path to save the model (recommended .keras extension)
        monitor='val_loss',           # Metric to monitor for improvement (e.g., 'val_loss' or 'val_accuracy')
        save_best_only=True,          # Save only when the monitored metric improves
        mode='min',                   # 'min' for loss, 'max' for accuracy
        verbose=1                     # Display messages when a new best model is saved
    )
# Create a learning rate schedule
lr_scheduler = custom_schedule(config.EMBED_DIM, warmup_steps=config.warmup_steps)
optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Compile the model
caption_model.compile(optimizer=optimizer, loss=cross_entropy)

# Fit the model
history = caption_model.fit(train_dataset,
                            epochs=config.EPOCHS,
                            validation_data=valid_dataset,
                            callbacks=[
                                early_stopping, 
                                tensorboard_callback,
                                checkpoint_callback
                                ])

print("# Compute definitive metrics on train/valid set")
train_metrics = caption_model.evaluate(train_dataset, batch_size=config.BATCH_SIZE)
valid_metrics = caption_model.evaluate(valid_dataset, batch_size=config.BATCH_SIZE)
print("Train Loss = %.4f - Train Accuracy = %.4f" % (train_metrics[0], train_metrics[1]))
print("Valid Loss = %.4f - Valid Accuracy = %.4f" % (valid_metrics[0], valid_metrics[1]))
if config.TEST_SET:
    test_metrics = caption_model.evaluate(test_dataset, batch_size=config.BATCH_SIZE)
    print("Test Loss = %.4f - Test Accuracy = %.4f" % (test_metrics[0], test_metrics[1]))

# Save training history under the form of a json file
history_dict = history.history
json.dump(history_dict, open(config.SAVE_DIR + 'history.json', 'w'))

# Save weights model
#caption_model.save_weights(config.SAVE_DIR + 'model.weights.h5')
caption_model.save_weights(config.get_model_weights_path)

# Save config model train
config_train = {"IMAGE_SIZE": config.IMAGE_SIZE,
                "MAX_VOCAB_SIZE" : config.MAX_VOCAB_SIZE,
                "SEQ_LENGTH" : config.SEQ_LENGTH,
                "EMBED_DIM" : config.EMBED_DIM,
                "NUM_HEADS" : config.NUM_HEADS,
                "FF_DIM" : config.FF_DIM,
                "BATCH_SIZE" : config.BATCH_SIZE,
                "EPOCHS" : config.EPOCHS,
                "VOCAB_SIZE" : VOCAB_SIZE,
                "CNN_MODEL" : config.cnnmodel
                }

json.dump(config_train, open(config.SAVE_DIR + 'config_train.json', 'w'))

# Save Tokenizer model
save_tokenizer(tokenizer, config.tokernizer_path)
