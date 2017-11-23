# Sentiment Classification with Differentiable Neural Computers
The scope of this work is to perform a complex task like sentiment classification(cross domain and in-domain) with one of the newest type of recurrente neural network, like the DNC's.

In order to do this, i have used the TensorFlow' s [implementation](https://github.com/deepmind/dnc) provided by Google DeepMind 
and the [Amazon Datasets](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) provided by the University of Stanford. 
To achieve the best results i used a [pre-trained word2vec model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) to encode the reviews and pass them to the network.
# Instructions
To run this code you had to follow these simple steps, if you know Italian you can find a more precise instructions on [istruzioni.pdf](./istruzioni.pdf).
## Requirements
To run this code you should have installed these Python libraries:
- [Tensorflow 1.1](https://www.tensorflow.org/install/)
- [Sonnet](https://github.com/deepmind/sonnet#installation)
- Numpy
- Gensim
- NLTK (Natural Language ToolKit)

To install Numpy,Gensim and NLTK you can simply launch this command (assuming you use a linux distribution and the file requirements.txt is in the same folder where you are with the terminal):
```
sudo pip install -r requirements.txt
```
## Datasets
In my bachelor thesis i used both Amazon datasets and Stanford Sentiment Treebank, feel free to use whatever dataset you want, but remember that my utility function to collect the training and testing data works only on json file with at least these two fields:
- reviewText
- overall
### Amazon datasets
- [Books(B)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz)
- [Electronics(E)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz)
- [Clothing,Shoes and Jewelry(J)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz)
- [Movies and TV (M)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz)
### Stanford Sentiment Treebank
To use the SST with my project I had to make some adjustments, the resulting datasets are here:
- [Training data](https://drive.google.com/file/d/0B_c80gg7c8oIOTF2OXdLUm55VGM/view?usp=sharing)
- [Test data](https://drive.google.com/open?id=0B_c80gg7c8oISkJsbUlMMkhOSUE)
## Run an experiment
There are 4 different files, everyone with a different goal:
- SentimentClassificationCrossDomainOfficial.py
- SentimentClassificationOfficial.py
- SentimentClassificationStanford.py
- SentimentClassificationWithTitleOfficial.py
The general sintax to run an experiment is:
```
python file_name.py --configuration configuration_file.json
```
Where file_name.py is one the 4 python file presented before and configuration_file.json is the file to use for configuring the experiment,  for further information about the configuration file see [Configuration](README#Configuration)
### In-Domain Experiment without review title
To run an in-domain experiment:
- Specify the dataset to use updating the field 'dataset' of configuration_file.json
- Update all those configuration file fields that you think had to be changed
- Launch the experiment with this command:
```
python SentimentClassificationOfficial.py --configuration configuration_file.json
```
### In-Domain Experiment with review title
To run an in-domain experiment with review title:
- Specify the dataset to use updating the field 'dataset' of configuration_file.json
- Update all those configuration file fields that you think had to be changed
- Launch the experiment with this command:
```
python SentimentClassificationWithTitleOfficial.py --configuration configuration_file.json
```
### Cross-Domain Experiment
To run an in-domain experiment:
- Specify the dataset to use for training(source domain) updating the field 'dataset' and the dataset to use for test(target domain) updating the field 'dataset_dest' of configuration_file.json
- Update all those configuration file fields that you think had to be changed
- Launch the experiment with this command:
```
python SentimentClassificationCrossDomainOfficial.py --configuration configuration_file.json
```
### Stanford Sentiment Treebank Experiment
To run an in-domain experiment:
- In the 'dataset' field, enter the path to the StanfordSentencesNTest file.json and in the 'dataset_dest' field the path to the StanfordSentencesNTest.json
- Update all those configuration file fields that you think had to be changed
- Launch the experiment with this command:
```
python SentimentClassificationStanford.py --configuration configuration_file.json
```
## Configuration
```
{
"hidden_size": "256",
"memory_size": "32",
"word_size": "64",
"num_write_heads": "1",
"num_read_heads": "1",
"clip_value": "10",
"max_grad_norm": "10",
"batch_size":"60",
"learning_rate": "1e-3",
"final_learning_rate": "1e-3",
"optimizer_epsilon": "1e-10",
"num_training_iterations": "1620",
"num_testing_iterations": "420",
"num_epochs": "8",
"report_interval": "10",
"checkpoint_dir" :
"/home/diego/sentiment-classification/mega/mixed256",
"checkpoint_interval": "-1",
"word_dimension" : "300",
"max_lenght" : "150",
"dataset" :
"/media/diego/Volume/Reviews/reviews_Electronics.json",
"datasetDest" :
"/media/diego/Volume/Reviews/reviews_Electronics.json",
"w2v_model" :
"/media/diego/Volume/GoogleNews-vectors-negative300.bin",
"random" : "True",
"seed" : "19",
"num_classes": "2"
}
```
