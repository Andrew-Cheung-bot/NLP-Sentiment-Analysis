#  Sentiment Ayalysis

## Abstract
...

## 1. Introduction
...

## 2. Background Survey
...

## 3. Methodology

### 3.1 Vectorization Method

The computer can not directly process human language (i.e., human words or sentences), so we need a technique to map our words or sentences to a specific number that the computer can process. In 2013, an algorithm called Word2Vec has been published. It can map a word to a vector with a fixed dimension, and still keep the semantic meaning of the word. For example, we have the classic equation: 
$$
Vector_{king}-Vector_{man}+Vector_{woman}\approx Vector_{queen}
$$
With the keeping of semantic meaning, we can perfectly convert the human language to the word vector without losing too much information. The main algorithms to get this vector are Continuous Bag-Of-Words (CBOW) and continuously sliding skip-gram.

The algorithm to map a sentence to a fixed dimension vector is called Sent2Vec, which is similar to the Word2Vec.

In this report, we will use fastText as the Word2Vec model and paraphrase-multilingual-mpnet-base-v2 as the Sent2Vec model. 

> [**fastText**](https://github.com/facebookresearch/fastText/)  
> fastText is a library for efficient text classification and representation learning. It provides pre-trained models for 157 languages. We will leverage the German model in our report. The dimension of the word vector we choose is 300, which means each word will be represented with a 300-dimension vector. For the words that are not included in the Word2Vec model, we use the "unknown" word vector as the placeholder. Because some models need a fixed training data size, so we imputation each comment or text with the "end-of-sentence" vector to let all comments or texts have the same length.

> [**Sentence-BERT**](https://arxiv.org/abs/1908.10084)  
> Sentence-BERT is a state-of-the-art sentence embedding model. Here, we use its paraphrase-multilingual-mpnet-base-v2 version. It can map the sentences & paragraphs to a 768-dimensional dense vector space.

### 3.2 Support Vector Machine
...


### 3.3 Vanilla RNN (or GRU/LSTM)

Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) that was introduced by Cho et al. in 2014 as a simpler alternative to Long Short-Term Memory (LSTM) networks. Like LSTM, GRU can process sequential data such as text, speech, and time-series data.

we will utilize the basic RNN cell that processes input sequences one element at a time while maintaining an internal state. This allows the network to remember past information and make predictions based on the sequence context.

### 3.4 Encoder-Only Transformer

The Transformer architecture was proposed in 2017 by google. It contains two parts: the encoder module and the decoder module. We use encoder-only architecture because this design can better understand the sentiment of the input text.  

_The hyperparameters setting is by doing grid searching:_ 
for nhead in $\{ 1,2,4\}$;  
for dim_feedforward in $\{4, 8, 16, 32 \}$;  
for nun_layer of encoders in $\{2, 4, 6 \}$;  
for dropout rate in $\{ 0.1, 0.3 \}$.

### 3.5 BERT-Based Classifier

The BERT is a transformers model pre-trained on a large corpus of multilingual data in a self-supervised fashion. The model used Masked language modeling (MLM) and Next sentence prediction (NSP), allowing it to extract the bidirectional representation from the input sentences. 

In this report, we will leverage the bert-base-multilingual-cased, which supports up to 104 languages. We will use the corresponding tokenizer to get the tokens for the training data rather than the tokenizer mentioned above. 

## 4. Experiments
- [German Dataset](https://github.com/UCSM-DUE/IWG_hatespeech_public) __(with Expert 1's labels)__

|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |  0.2872  |  0.2796   | 1.0000 |  0.4370  |
|           BERT            |  0.7234  |  0.6948   | 0.7234 |  0.6999  |
|       Standard GRU Net    |  0.7376  |   0.5     | 0.3784 |  0.4308  |
- [English Dataset (SOCC)](https://www.kaggle.com/datasets/mtaboada/sfu-opinion-and-comments-corpus-socc) 

|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |          |           |        |          |
|           BERT            |  0.8086  |  0.7663   | 0.8086 |  0.7724  |
|       Standard GRU Net    |  0.7636  |  0.3667   | 0.3793 |  0.3729  |

## 5. Application



## References
