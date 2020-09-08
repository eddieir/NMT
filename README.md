This repo is dedicated to the implementation of Neural machine translation(NMT) encoder-decoder with using of Keras. This model takes input sentences which taken from http://www.manythings.org/anki/ and translate it to the corresponsing sentence and the Bleu score which taken from the run on the test set is 0.509124.

This model has two main section which is Encoder and Decoder:

**Encoder** : This section refers to the input text corpus which is German text in the form of embedding vectors and trains the model. 

**Decoder**:  This part of the model translates ands predicts the input embedding vectors into one-hot vectors which respresents English words in the dictionary. 

![model](https://user-images.githubusercontent.com/23243761/92496617-a58ebd00-f1f8-11ea-9ccb-056ec52dafcd.png)

**Requirements**:
For this projects it requried to have the following python packages: Numpy,nltk and kers beside all of them it required to install tensorflow because without it Keras will not work. 

In terms of the dataset which used taken from http://www.manythings.org/anki/ and once you entered to this URL there are bunch texts you need to download the deu.txt which contain the pairs of English-German sentences. And this dataset contains 1,52,820 pairs of English to German phrases. 

The first phase of this project is dedicating to preprocessing of the dataset, so in order to do that you need to first run the pre-process.py in order to clean the data and then run prepare_dataset.py to break the dataset into smaller trainig and testing dataset. Once this two python ran they will generate three pickle file which are english-german-both.pkl, english-german-train.pkl and english-german-test.pkl.

The preprocessing of the data involves:

    Removing punctuation marks from the data.
    Converting text corpus into lower case characters.
    Shuffling the sentences as sentences were previously sorted in the increasing order of their length.

**Training the Encoder-Decoder LSTM model**

Run model.py to train the model. After successful training, the model will be saved as model.h5 in your current directory.

    This model uses Encoder-Decoder LSTMs for NMT. In this architecture, the input sequence is encoded by the front-end model called encoder then, decoded by backend model called decoder.
    It uses Adam Optimizer to train the model using Stochastic Gradient Descent and minimizes the categorical loss function.

**Evaluating the model**

Run evaluate_model.py to evaluate the accuracy of the model on both train and test dataset.

    It loads the best saved model.h5 model.
    The model performs pretty well on train set and have been generalized to perform well on test set.
    After prediction, we calculate Bleu scores for the predicted sentences to check how well the model generalizes.

**Calculating the Bleu scores**

BLEU (bilingual evaluation understudy) is an algorithm for comparing predicted machine translated text with the reference string given by the human. A high BLEU score means the predicted translated sentence is pretty close to the reference string. More information can be found here. Below are the BLEU scores for both the training set and the testing set along with the predicted and target English sentence corresponding to the given German source sentence.
