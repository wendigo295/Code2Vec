## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/wendigo295/Code2Vec.git
cd code2vec
```

### Step 1: Creating a new dataset from java sources
In order to have a preprocessed dataset to train a network on, you can either download our
preprocessed dataset, or create a new dataset of your own.

#### Download preprocessed dataset of ~14M examples (compressed: 6.3GB, extracted 32GB)
```
wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
tar -xvzf java14m_data.tar.gz
```
This will create a data/java14m/ sub-directory, containing the files that hold the training, test and validation sets,
and a vocabulary file for various dataset properties.

#### Creating and preprocessing a new Java dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh

### Step 2: Training a model
You can either download an already-trained model, or train a new model using a preprocessed dataset.

#### Downloading a trained model (1.4 GB)
We already trained a model for 8 epochs on the data that was preprocessed in the previous step.
The number of epochs was chosen using [early stopping](https://en.wikipedia.org/wiki/Early_stopping), as the version that maximized the F1 score on the validation set. This model can be downloaded [here](https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz) or using:
```
wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz
tar -xvzf java14m_model.tar.gz
```

##### Note:
This trained model is in a "released" state, which means that we stripped it from its training parameters and can thus be used for inference, but cannot be further trained. If you use this trained model in the next steps, use 'saved_model_iter8.release' instead of 'saved_model_iter8' in every command line example that loads the model such as: '--load models/java14_model/saved_model_iter8'. To read how to release a model, see [Releasing the model](#releasing-the-model).

#### Downloading a trained model (3.5 GB) _which can be further trained_

A non-stripped trained model can be obtained [here](https://s3.amazonaws.com/code2vec/model/java14m_model_trainable.tar.gz) or using:

```
wget https://s3.amazonaws.com/code2vec/model/java14m_model_trainable.tar.gz
tar -xvzf java14m_model_trainable.tar
```  

This model weights more than twice than the stripped version, and it is recommended only if you wish to continue training a model which is already trained. To continue training this trained model, use the `--load` flag to load the trained model; the `--data` flag to point to the new dataset to train on; and the `--save` flag to provide a new save path.

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to our "java14m" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [config.py](config.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:
```
source train.sh
```

##### Notes:
  1. By default, the network is evaluated on the validation set after every training epoch.
  2. The newest 10 versions are kept (older are deleted automatically). This can be changed, but will be more space consuming.
  3. By default, the network is training for 20 epochs.
These settings can be changed by simply editing the file [config.py](config.py).

### Step 3: Evaluating a trained model
Once the score on the validation set stops improving over time, you can stop the training process (by killing it)
and pick the iteration that performed the best on the validation set.
Suppose that iteration #8 is our chosen model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --test data/java14m/java14m.test.c2v
```
While evaluating, a file named "log.txt" is written with each test example name and the model's prediction.

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --predict
```
After the model loads, follow the instructions and edit the file [Input.java](Input.java) and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.

## Configuration
Changing hyper-parameters is possible by editing the file
[config.py](config.py).

Here are some of the parameters and their description:
#### config.NUM_TRAIN_EPOCHS = 20
The max number of epochs to train the model. Stopping earlier must be done manually (kill).
#### config.SAVE_EVERY_EPOCHS = 1
After how many training iterations a model should be saved.
#### config.TRAIN_BATCH_SIZE = 1024 
Batch size in training.
#### config.TEST_BATCH_SIZE = config.TRAIN_BATCH_SIZE
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10
Number of words with highest scores in $ y_hat $ to consider during prediction and evaluation.
#### config.NUM_BATCHES_TO_LOG_PROGRESS = 100
Number of batches (during training / evaluating) to complete between two progress-logging records.
#### config.NUM_TRAIN_BATCHES_TO_EVALUATE = 100
Number of training batches to complete between model evaluations on the test set.
#### config.READER_NUM_PARALLEL_BATCHES = 4
The number of threads enqueuing examples to the reader queue.
#### config.SHUFFLE_BUFFER_SIZE = 10000
Size of buffer in reader to shuffle example within during training.
Bigger buffer allows better randomness, but requires more amount of memory and may harm training throughput.
#### config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
The buffer size (in bytes) of the CSV dataset reader.

#### config.MAX_CONTEXTS = 200
The number of contexts to use in each example.
#### config.MAX_TOKEN_VOCAB_SIZE = 1301136
The max size of the token vocabulary.
#### config.MAX_TARGET_VOCAB_SIZE = 261245
The max size of the target words vocabulary.
#### config.MAX_PATH_VOCAB_SIZE = 911417
The max size of the path vocabulary.
#### config.DEFAULT_EMBEDDINGS_SIZE = 128
Default embedding size to be used for token and path if not specified otherwise.
#### config.TOKEN_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for tokens.
#### config.PATH_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for paths.
#### config.CODE_VECTOR_SIZE = config.PATH_EMBEDDINGS_SIZE + 2 * config.TOKEN_EMBEDDINGS_SIZE
Size of code vectors.
#### config.TARGET_EMBEDDINGS_SIZE = config.CODE_VECTOR_SIZE
Embedding size for target words.
#### config.MAX_TO_KEEP = 10
Keep this number of newest trained versions during training.
#### config.DROPOUT_KEEP_RATE = 0.75
Dropout rate used during training.
#### config.SEPARATE_OOV_AND_PAD = False
Whether to treat `<OOV>` and `<PAD>` as two different special tokens whenever possible.

## Features
Code2vec supports the following features: 

### Releasing the model
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes 3x less disk space.

### Exporting the trained token vectors and target vectors
Token and target embeddings are available to download: 

[[Token vectors]](https://s3.amazonaws.com/code2vec/model/token_vecs.tar.gz) [[Method name vectors]](https://s3.amazonaws.com/code2vec/model/target_vecs.tar.gz)

These saved embeddings are saved without subtoken-delimiters ("*toLower*" is saved as "*tolower*").

In order to export embeddings from a trained model, use the "--save_w2v" and "--save_t2v" flags:

Exporting the trained *token* embeddings:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --save_w2v models/java14_model/tokens.txt
```
Exporting the trained *target* (method name) embeddings:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --save_t2v models/java14_model/targets.txt
```
This saves the tokens/targets embedding matrices in word2vec format to the specified text file, in which:
the first line is: \<vocab_size\> \<dimension\>
and each of the following lines contains: \<word\> \<float_1\> \<float_2\> ... \<float_dimension\>

These word2vec files can be manually parsed or easily loaded and inspected using the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) python package:
```python
python3
>>> from gensim.models import KeyedVectors as word2vec
>>> vectors_text_path = 'models/java14_model/targets.txt' # or: `models/java14_model/tokens.txt'
>>> model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
>>> model.most_similar(positive=['equals', 'to|lower']) # or: 'tolower', if using the downloaded embeddings
>>> model.most_similar(positive=['download', 'send'], negative=['receive'])
```
The above python commands will result in the closest name to both "equals" and "to|lower", which is "equals|ignore|case".
Note: In embeddings that were exported manually using the "--save_w2v" or "--save_t2v" flags, the input token and target words are saved using the symbol "|" as a subtokens delimiter ("*toLower*" is saved as: "*to|lower*"). In the embeddings that are available to download (which are the same as in the paper), the "|" symbol is not used, thus "*toLower*" is saved as "*tolower*".

### Exporting the code vectors for the given code examples
The flag `--export_code_vectors` allows to export the code vectors for the given examples. 

If used with the `--test <TEST_FILE>` flag,
a file named `<TEST_FILE>.vectors` will be saved in the same directory as `<TEST_FILE>`. 
Each row in the saved file is the code vector of the code snipped in the corresponding row in `<TEST_FILE>`.
 
If used with the `--predict` flag, the code vector will be printed to console.

