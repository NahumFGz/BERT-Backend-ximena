# -*- coding: utf-8 -*-
dataset_name  = 'zdataset_tres_clases_2'  # [zdataset_tres_clases_2, DATD_all_dos_clases]
balanceo      = 'normal'                  # [normal, over]
batch_size    = 32                        # [8, 16, 32]
test_sz       = 0.1                       # [0.1, 0.2, 0.3]
learning_rate = 3e-5                      # [5e-5, 4e-5, 3e-5, 2e-5]
num_epochs    = 10                        # [2,3,4,5,6,7,8,9,10]
flg_preproc   = 'processed'               # ['original', 'processed']
model         = 'base-uncased'
rand_state    = 2020

"""# B - Setup"""
import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    #print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    #print('Device name:', torch.cuda.get_device_name(0))

else:
    #print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# SpanBERT https://github.com/chriskhanhtran/spanish-bert
# Beto https://github.com/dccuchile/beto
# Multilingual https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages

dataset_path  = '/content/drive/MyDrive/DataSets/Tesis_Ximena_2/'+ dataset_name +'.csv'
result_dir   =  '/content/drive/MyDrive/ResultadosEntrenamiento/Tesis_Ximena_2/'+ dataset_name + '_PROC'+ flg_preproc  + '_MDL'+ model + '_BLC'+ balanceo +'_TS'+ str(test_sz) + '_RS'+ str(rand_state) +'_epch' + str(num_epochs) + '_lr' + str(learning_rate)

if model == 'base-uncased':
  path_model = 'bert-base-uncased'
elif model == 'SpanBERT':
  path_model = 'dccuchile/bert-base-spanish-wwm-uncased'
elif model == 'BETO':
  path_model = 'mrm8488/bert-spanish-cased-finetuned-ner'
elif model == 'multilingual':
  path_model = 'bert-base-multilingual-uncased'
#print(path_model)

"""## 1. Load Essential Libraries"""

# Commented out IPython magic to ensure Python compatibility.
#https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
import os
import re
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

"""## 2. Dataset"""

#from google.colab import drive
#drive.mount('/content/drive')

data = pd.read_csv(dataset_path, encoding= 'unicode_escape')
data.head(5)

len(data.index)

"""### 2.1. Train - Test split"""

from sklearn.model_selection import train_test_split
X = data.text.values
y = data.label.values
X_TRAIN, X_test, y_TRAIN, y_test = train_test_split(X, y, test_size=test_sz, random_state=rand_state)
#X_TRAIN, X_test, y_TRAIN, y_test = train_test_split(X, y, test_size=test_sz, random_state=rand_state, stratify=y)

from collections import Counter
counter = Counter(y_TRAIN)
counter

from collections import Counter
counter = Counter(y_test)
counter

"""#### Extra: Balanceo"""

# Unir
np_train = []
for i in range(X_TRAIN.shape[0]):
  np_train.append([X_TRAIN[i],y_TRAIN[i]])

# Aquí le aplicamos el balanceo
df_TRAIN  = pd.DataFrame(np_train, columns=['text','label'])
val_count = df_TRAIN['label'].value_counts()

# Crear arrays
np_class_0 = df_TRAIN[df_TRAIN['label']==0].values.tolist()
np_class_1 = df_TRAIN[df_TRAIN['label']==1].values.tolist()
np_class_2 = df_TRAIN[df_TRAIN['label']==2].values.tolist()

# Pintar valores
#print('0: ' + str(len(np_class_0)))
#print('1: ' + str(len(np_class_1)))
#print('2: ' + str(len(np_class_2)))

np.random.seed(rand_state)
np_TRAIN = []
# Normal
if balanceo == 'normal':
  np_TRAIN = np_class_0 + np_class_1 + np_class_2

dfTRAIN = pd.DataFrame(np_TRAIN, columns=['text','label'])
dfTRAIN['label'].value_counts()

X_TRAIN = dfTRAIN.text.values
y_TRAIN = dfTRAIN.label.values

"""### 2.2. Train - valid Split"""

X_train, X_val, y_train, y_val = train_test_split(X_TRAIN, y_TRAIN, test_size=test_sz, random_state=rand_state)

#print('All dataset: ', data.shape[0])
#print('Train: ', X_train.shape[0])
#print('Valid: ', X_val.shape[0])
#print('Test: ',+ X_test.shape[0])
#print('Train + valid + test:', str(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))

import numpy as np

def plot_confusion_matrix(name_flag,
                          cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(result_dir +'_' + name_flag +'_CM.png')
    plt.show()

from sklearn.metrics import accuracy_score, roc_curve, auc,classification_report,confusion_matrix

def evaluate_roc(probs, y_true, name_flag):
  # Get predictions from the probabilities
  y_pred_list = []
  for pred in probs:
    if np.argmax(pred) == 0:
      y_pred_list.append(0)  
    elif np.argmax(pred) == 1:
      y_pred_list.append(1)    
    elif np.argmax(pred) == 2:
      y_pred_list.append(2)  
  y_pred = np.array(y_pred_list)

  # Guardar predicción y test
  probs_test = np.append(probs, np.transpose([y_pred]), axis = 1)
  probs_test = np.append(probs_test, np.transpose([y_test]), axis = 1)
  df = pd.DataFrame(probs_test, columns=['probs_x','probs_y','probs_z', 'y_pred','y_true'])
  df.to_csv(result_dir +'_' +  name_flag +'_History.csv', index=False)

  # Calcular la precisión
  accuracy = accuracy_score(y_true, y_pred)
  print(f'Accuracy: {accuracy*100:.2f}%')
  
  plot_confusion_matrix(name_flag,
                        confusion_matrix(y_true, y_pred),
                        target_names = ['Class 0', 'Class 1', 'Class 2'],
                        title='Confusion matrix',
                        cmap=None,
                        normalize=False
                        )

# Evaluate the Bert classifier
#evaluate_roc(probs_test, y_test,'test')

"""# D - Fine-tuning BERT

## 1. Install the Hugging Face Library
"""

#!pip install transformers==4.4.2
#2.8.0

"""# 2. Tokenization and Input Formatting"""

#https://pypi.org/project/tweet-preprocessor/
#!pip install tweet-preprocessor

import preprocessor as p

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    if flg_preproc == 'processed':
      # Remove '@name'
      text = re.sub(r'(@.*?)[\s]', ' ', text)

      # Replace '&amp;' with '&'
      text = re.sub(r'&amp;', '&', text)

      # Replace '&amp' with '&'
      text = re.sub(r'&amp', '&', text)

      # Replace " with ''
      text = re.sub(r'"', '', text)

      # Remove trailing whitespace
      text = re.sub(r'\s+', ' ', text).strip()
      
      # Replace HashTags
      text = re.sub(r'#', '', text)

      # Preprocesamiento de https://pypi.org/project/tweet-preprocessor/
      text = p.clean(text)

    return text

#print(p.clean('@RT @Twitter raw text data usually has lots of #residue. http://t.co/g00gl'))
#print(text_preprocessing('@RT @Twitter raw text data usually has lots of #residue. http://t.co/g00gl'))

#print(p.clean('@user im depression'))
#print(text_preprocessing('@user im depression'))

# Print sentence 0
#print('Original: ', X_train[0])
#print('Processed: ', text_preprocessing(X_train[0]))

"""### 2.1. BERT Tokenizer"""

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(path_model, do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

"""Before tokenizing, we need to specify the maximum length of our sentences."""

# Concatenate train data and test data
all_tweets = np.concatenate([X_train, X_test, X_val])

# Encode our concatenated data
encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]

# Find the maximum length
max_len = max([len(sent) for sent in encoded_tweets])
#print('Max length: ', max_len)

"""Now let's tokenize our data."""

# Specify `MAX_LEN`
MAX_LEN = max_len

# Print sentence 0 and its encoded token ids
token_ids = list(preprocessing_for_bert([X_train[0]])[0].squeeze().numpy())
#print('Original: ', X_train[0])
#print('Token IDs: ', token_ids)

# Run function `preprocessing_for_bert` on the train set and the validation set
#print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)

"""### 2.2. Create PyTorch DataLoader"""

#print(train_inputs.shape)
#print(val_inputs.shape)

"""We will create an iterator for our dataset using the torch DataLoader class. This will help save on memory during training and boost the training speed."""

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
# batch_size = 32

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

"""## 3. Train Our Model

### 3.1. Create BertClassifier
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
import torch.nn as nn
from transformers import BertModel

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 3

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(path_model)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

"""### 3.2. Optimizer & Learning Rate Scheduler

To fine-tune our Bert Classifier, we need to create an optimizer. The authors recommend following hyper-parameters:

- Batch size: 16 or 32
- Learning rate (Adam): 5e-5, 3e-5 or 2e-5
- Number of epochs: 2, 3, 4

Huggingface provided the [run_glue.py](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109) script, an examples of implementing the `transformers` library. In the script, the AdamW optimizer is used.
"""

from transformers import AdamW, get_linear_schedule_with_warmup

def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=learning_rate,         #5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

"""### 3.3. Training Loop"""

import random
import time

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    # Start training loop
    #print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        #print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        #print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                #print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        #print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            #print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            #print("-"*70)
        #print("\n")
    
    #print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

"""Now, let's start training our BertClassifier!"""

set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=num_epochs)
#train(bert_classifier, train_dataloader, val_dataloader, epochs=num_epochs, evaluation=True)

"""### 3.4. Evaluation on Validation Set"""

import torch.nn.functional as F

def bert_predict(model, test_dataloader):
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

# Compute predicted probabilities on the test set
probs = bert_predict(bert_classifier, val_dataloader)

# Evaluate the Bert classifier
# evaluate_roc(probs, y_val,'valid')

"""# 3.5. Train Our Model on the Entire Training Data"""

# Concatenate the train set and the validation set
full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
full_train_sampler = RandomSampler(full_train_data)
full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)

# Train the Bert Classifier on the entire training data
set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=num_epochs)
train(bert_classifier, full_train_dataloader, epochs=num_epochs)

"""## 4. Predictions on Test Set

### 4.1. Data Preparation

Before making predictions on the test set, we need to redo processing and encoding steps done on the training data. Fortunately, we have written the `preprocessing_for_bert` function to do that for us.
"""

# Run `preprocessing_for_bert` on the test set
#print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(X_test)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

"""# 4.2. **PREDICCION**"""

# Compute predicted probabilities on the test set
probs_test = bert_predict(bert_classifier, test_dataloader)

# Number of tweets predicted non-negative
#print("Number of tweets predicted non-negative: ", preds_test.sum())
#print(result_dir)

# Evaluate the Bert classifier
evaluate_roc(probs_test, y_test,'test')

"""# E - Conclusion

## Guardar Modelo
"""
try:
  PATH_SAVE_MODEL_1 =  result_dir + '_model1.pth' 
  torch.save(bert_classifier, PATH_SAVE_MODEL_1)
except:
  print('Fin...!!!')
