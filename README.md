#  <div align="center">ML Techniques for Cross-Lingual Cyberbullying Detection</div>

## Abstract
Cyberbullying, a pervasive issue in the digital age, necessitates robust detection mechanisms to safeguard individuals across various online platforms. Many scholars have applied various techniques for cyberbullying detection. This study contributes to the field by evaluating the effectiveness of advanced machine learning techniques in detecting cyberbullying on German and English. We employed three distinct computational models: Support Vector Machine (SVM), Recurrent Neural Network (RNN), and BERT, to assess their performance in identifying cyberbullying incidents. We use metrics such as precision, recall, and F1-score to gauge the models' detection capabilities. Additionally, we explored the adaptability of these models through cross-lingual transfer learning experiments, assessing their ability to generalize from one language to another. 

## 1. Background Survey
Cyberbullying<sup><a href="#ref1">1</a></sup>, often described as sharing insulting or embarrassing comments, photos, or videos online, is becoming more common on social networks<sup><a href="#ref2">2</a></sup>. The increase in cyberbullying and its connection to serious issues like depression, low self-esteem, and suicidal thoughts and behaviors have made it a major public health issue<sup><a href="#ref3">3,4</a></sup>. This has also sparked a lot of research in both psychology and computer science to better understand cyberbullying's characteristics and its impact on social networks. Research on cyberbullying can provide valuable insights and promote self-reflection on people's behaviour<sup><a href="#ref5">5</a></sup>.One of the primary challenges in the  detection of cyberbullying is determining the occurrence of such behavior. Rosa H. et al<sup><a href="#ref6">6</a></sup>. present a systematic review of existing research, concentrating on four main criteria based on the established definitions of cyberbullying. These criteria include: a) the use of aggressive or hostile language, b) the intention to harm others, c) the repetitive nature of the behavior, and d) the occurrence among peers. The current academic approaches to cyberbullying detection can be categorized into: rule-based methods, machine learning methods, and deep learning methods.  

Rule-based approaches to cyberbullying detection involve designing systems that use predefined rules to identify harmful content<sup><a href="#ref6">6</a></sup>. These systems typically utilize methods such as subjectivity analysis, lexical syntactic features<sup><a href="#ref7">7</a></sup>, and curse word dictionaries<sup><a href="#ref8">8</a></sup> to assess the offensiveness of content and predict whether a user exhibits bullying behavior. While these approaches can achieve high accuracy under specific conditions, they are often criticized for their lack of adaptability and inability to generalize to new situations or consider the broader context of social media interactions<sup><a href="#ref7">7</a></sup>.  

To date, machine learning models are the most widely applied methods. Classifiers such as Support Vector Machines (SVM), Naive Bayes (NB), Logistic Regression (LR), and Gradient Boosting .etc have been extensively tested across various social media platforms<sup><a href="#ref6">6</a></sup>.
Due to the subjective nature of bullying expressions, traditional ML models perform less effectively in detecting cyber harassment than deep learning (DL)-based methods. Recent research indicates that DL models surpass traditional ML algorithms in identifying cyberbullying<sup><a href="#ref9">9</a></sup>. The adoption of DL over traditional ML reflects a broader trend towards more sophisticated, automated solutions that can better understand the context and nuances of online interactions, thereby improving detection rates and adaptability across different social media environments.Deep Neural Networks, including Recurrent Neural Network (RNN), Gated Recurrent Unit (GRU)<sup><a href="#ref10">10</a></sup>, Long Short-Term Memory (LSTM)<sup><a href="#ref11">11</a></sup>, and various other deep learning models, can be utilized for detecting this issue.  

With the advent of large language models, a new path has been opened for cyberbullying detection. Models like Transformers<sup><a href="#ref12">12</a></sup> and BERT<sup><a href="#ref13">13</a></sup>, which have demonstrated effectiveness in detecting cyberbullying, allow researchers to achieve impressive performance in identifying and preventing cyberbullying behaviors in social media content by utilizing finely-tuned pretrained large language models.

## 2. Methodology

### 2.1 Vectorization Method

The computer can not directly process human language (i.e., human words or sentences), so we need a technique to map our words or sentences to a specific number that the computer can process. In 2013, an algorithm called Word2Vec has been published. It can map a word to a vector with a fixed dimension, and still keep the semantic meaning of the word.  
For example, we have the classic equation: 

$$Vector_{king}-Vector_{man}+Vector_{woman}\approx Vector_{queen}$$

With the keeping of semantic meaning, we can perfectly convert the human language to the word vector without losing too much information. The main algorithms to get this vector are Continuous Bag-Of-Words (CBOW) and continuously sliding skip-gram.

The algorithm to map a sentence to a fixed dimension vector is called Sent2Vec, which is similar to the Word2Vec.

In this report, we will use fastText as the Word2Vec model and paraphrase-multilingual-mpnet-base-v2 as the Sent2Vec model. 

> [**fastText**](https://github.com/facebookresearch/fastText/)  
> fastText is a library for efficient text classification and representation learning. It provides pre-trained models for 157 languages. We will leverage the German model in our report. The dimension of the word vector we choose is 300, which means each word will be represented with a 300-dimension vector. For the words that are not included in the Word2Vec model, we use the "unknown" word vector as the placeholder. Because some models need a fixed training data size, so we imputation each comment or text with the "end-of-sentence" vector to let all comments or texts have the same length.

> [**Sentence-BERT**](https://arxiv.org/abs/1908.10084)  
> Sentence-BERT is a state-of-the-art sentence embedding model. Here, we use its paraphrase-multilingual-mpnet-base-v2 version. It can map the sentences & paragraphs to a 768-dimensional dense vector space.

### 2.2 Support Vector Machine
Support Vector Machines (SVM) are a robust class of supervised learning algorithms used primarily for classification and regression tasks. At its core, SVM constructs a hyperplane or set of hyperplanes in a high-dimensional space, which can be used for pattern recognition, outlier detection, and other complex data mining tasks. There have been a number of previous studies using SVMs for cyberbullying detection<sup><a href="#ref14">14</a></sup>.

### 2.3 Vanilla RNN (or GRU/LSTM)

Gated Recurrent Unit (GRU)<sup><a href="#ref15">15</a></sup> is a type of recurrent neural network (RNN) that was introduced by Cho et al. in 2014 as a simpler alternative to Long Short-Term Memory (LSTM) networks. Like LSTM, GRU can process sequential data such as text, speech, and time-series data.

we will utilize the basic RNN cell that processes input sequences one element at a time while maintaining an internal state. This allows the network to remember past information and make predictions based on the sequence context.

### 2.4 Encoder-Only Transformer

The Transformer architecture was proposed in 2017 by google<sup><a href="#ref16">16</a></sup>. It contains two parts: the encoder module and the decoder module. We use encoder-only architecture because this design can better understand the sentiment of the input text.  

_The hyperparameters setting is by doing grid searching:_  
for nhead in $\{ 1,2,4\}$;  
for dim_feedforward in $\{4, 8, 16, 32 \}$;  
for nun_layer of encoders in $\{2, 4, 6 \}$;  
for dropout rate in $\{ 0.1, 0.3 \}$.

### 2.5 BERT-Based Classifier

The BERT<sup><a href="#ref17">17</a></sup> is a transformers model pre-trained on a large corpus of multilingual data in a self-supervised fashion. The model used Masked language modeling (MLM) and Next sentence prediction (NSP), allowing it to extract the bidirectional representation from the input sentences. 

In this report, we will leverage the bert-base-multilingual-cased, which supports up to 104 languages. We will use the corresponding tokenizer to get the tokens for the training data rather than the tokenizer mentioned above. 

## 3. Experiments
We chose two datasets for our experiments, one is German Twitter Data Set <sup><a href="#ref18">18</a></sup> and the other is SFU Opinion and Comment Corpus <sup><a href="#ref19">19</a></sup>. We first trained and tested using each of the four methods on the German dataset, and the experimental results are shown below.

- [German Dataset](https://github.com/UCSM-DUE/IWG_hatespeech_public) __(with Expert 1's labels)__
  
|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |  0.6915  |  0.3636   | 0.1538 |  0.2162  |
|           BERT            |  0.7340  |  0.6000   | 0.1154 |  0.1935  |
|       Standard GRU Net    |  0.7376  |  0.5000   | 0.3784 |  0.4308  |
|           SVM             |  0.7234  |  0.6313   | 0.5831 |  0.5871  |

In evaluating the performance of four machine learning models on a cyberbullying detection task, we focused on four key metrics: Accuracy, Precision, Recall, and F1-Score. These metrics are essential for assessing the efficacy of models in classification tasks.  

Accuracy indicates the overall prediction correctness of the models. The Standard GRU Network showed the highest accuracy at 73.76%, indicating strong predictive power across samples. BERT and SVM models also demonstrated high accuracy rates at 73.40% and 72.34%, respectively, while the Encoder-Only model lagged at 69.15%.  

Precision measures the proportion of actual positives among those predicted as positive. The SVM model excelled with a precision of 63.125%, indicating high reliability in its positive predictions. The BERT model also performed well in precision at 60.00%. In contrast, the Standard GRU Network and Encoder-Only model had lower precision rates of 50.00% and 36.36%, respectively.  

Recall reflects the percentage of actual positive instances correctly identified. The SVM model again performed robustly with a recall rate of 58.31%, effectively capturing instances of cyberbullying. The Standard GRU Network also had a relatively high recall of 37.84%, whereas BERT and Encoder-Only models had significantly lower recalls at 11.54% and 15.38%, respectively.  

F1-Score, which is the harmonic mean of Precision and Recall, provides a balance between these metrics and is crucial for evaluating overall model performance. The SVM model led with the highest F1-Score of 58.71%, demonstrating a good balance between precision and recall. The Standard GRU Network had an F1-Score of 43.08%, and both BERT and Encoder-Only models had lower F1-Scores at 19.35% and 21.62%, respectively.

- [English Dataset (SOCC)](https://www.kaggle.com/datasets/mtaboada/sfu-opinion-and-comments-corpus-socc)

|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |  0.7943  |  0.3333   | 0.1026 |  0.1569  |
|           BERT            |  0.8134  |  0.5000   | 0.1795 |  0.2642  |
|       Standard GRU Net    |  0.7636  |  0.3667   | 0.3793 |  0.3729  |
|           SVM             |  0.7751  |  0.5736   | 0.5456 |  0.5489  |

BERT achieved the highest accuracy at 81.34%, indicating a strong overall predictive performance. It was followed by the Encoder-Only model with an accuracy of 79.43%, the SVM model at 77.51%, and the Standard GRU Network at 76.36%, which showed the lowest accuracy among the models.  

In terms of precision, which reflects the accuracy of positive predictions, SVM led with a precision rate of 57.36%, suggesting high reliability in its positive classifications. BERT exhibited a precision of 50.00%, while the Standard GRU Network and Encoder-Only model showed lower precision rates of 36.67% and 33.33%, respectively.  

Recall metrics, indicating the model's ability to identify actual positive instances, showed SVM with a strong performance at 54.56%. The Standard GRU Network had a recall of 37.93%, significantly higher than BERT's 17.95%. The Encoder-Only model had the lowest recall at 10.26%, indicating a struggle in capturing positive cases effectively.  

The F1-Score, which balances precision and recall, was highest for SVM at 54.89%, demonstrating a good equilibrium between identifying true positives and minimizing false positives. The Standard GRU Network followed with an F1-Score of 37.29%. BERT's F1-Score stood at 26.42%, while the Encoder-Only model had the lowest at 15.69%.  

In conclusion, we can see that SVM achieves good results in both datasets.SVM's superior performance in both datasets can largely be attributed to its ability to handle high-dimensional data effectively, making it well-suited for text classification tasks. Its design principle of maximizing the margin between classes helps ensure a robust separation, reducing generalization errors. In addition, it may be that the total amount of our training data is still relatively small, and thus the performance on the remaining three methods is not as good as SVM.Thus, it can be seen that SVM is still quite reliable and useful as a traditional machine learning method.

## 4. Application  

We collected 50 comments from Twitter at random and manually labeled each as either cyberbullying or non-bullying. We utilized this authentic data to evaluate the performance of our pre-trained models on English and German datasets. The experimental findings are presented in below.  

- Test results of pre-training the model on the German dataset

|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |          |           |        |          |
|           BERT            |          |           |        |          |
|       Standard GRU Net    |  0.6432  |  0.5152   | 0.1954 |  0.2833  |
|           SVM             |  0.52    |  0.755    | 0.52   |  0.37    |

- Test results of pre-training the model on the English dataset

|           model           | Accuracy | Precision | Recall | F1-score |
| :-----------------------: | :------: | :-------: | :----: | :------: |
|       Encoder-Only        |          |           |        |          |
|           BERT            |          |           |        |          |
|       Standard GRU Net    |  0.7163  |  0.3958   | 0.4578 |  0.4246  |
|           SVM             |  0.6     |  0.611    | 0.6    |  0.58    |

## References

<p id="ref1">[1]Feinberg, T.; Robey, N. Cyberbullying. Educ. Dig. 2009, 74, 26.</p>  
<p id="ref2">[2]Cheng L, Li J, Silva Y N, et al. Xbully: Cyberbullying detection within a multi-modal context[C]//Proceedings of the twelfth acm international conference on web search and data mining. 2019: 339-347.</p>  
<p id="ref3">[3]Nikolaou D. Does cyberbullying impact youth suicidal behaviors?[J]. Journal of health economics, 2017, 56: 30-46.</p>  
<p id="ref4">[4]Brailovskaia J, Teismann T, Margraf J. Cyberbullying, positive mental health and suicide ideation/behavior[J]. Psychiatry research, 2018, 267: 240-242.</p>   
<p id="ref5">[5]Van Royen K, Poels K, Vandebosch H, et al. “Thinking before posting?” Reducing cyber harassment on social networking sites through a reflective message[J]. Computers in human behavior, 2017, 66: 345-352.</p>   
<p id="ref6">[6]Yi P, Zubiaga A. Session-based cyberbullying detection in social media: A survey[J]. Online Social Networks and Media, 2023, 36: 100250.</p>   
<p id="ref7">[7]Chen Y, Zhou Y, Zhu S, et al. Detecting offensive language in social media to protect adolescent online safety[C]//2012 international conference on privacy, security, risk and trust and 2012 international confernece on social computing. IEEE, 2012: 71-80.</p> 
<p id="ref8">[8]Bretschneider U, Wöhner T, Peters R. Detecting online harassment in social networks[J]. 2014.</p>   
<p id="ref9">[9]Buan T A, Ramachandra R. Automated cyberbullying detection in social media using an svm activated stacked convolution lstm network[C]//Proceedings of the 2020 4th International Conference on Compute and Data Analysis. 2020: 170-174.</p>   
<p id="ref10">[10]Cho, K.; Van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; Bengio, Y. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv 2014, arXiv:1406.1078.</p>   
<p id="ref11">[11]Gers F A, Schmidhuber J, Cummins F. Learning to forget: Continual prediction with LSTM[J]. Neural computation, 2000, 12(10): 2451-2471.</p>   
<p id="ref12">[12]Pericherla S, Ilavarasan E. Transformer network-based word embeddings approach for autonomous cyberbullying detection[J]. International Journal of Intelligent Unmanned Systems, 2024, 12(1): 154-166.</p>   
<p id="ref13">[13]Guo X, Anjum U, Zhan J. Cyberbully detection using bert with augmented texts[C]//2022 IEEE International Conference on Big Data (Big Data). IEEE, 2022: 1246-1253.</p>    
<p id="ref14">[14]Ali A, Syed A M. Cyberbullying detection using machine learning[J]. Pakistan Journal of Engineering and Technology, 2020, 3(2): 45-50.</p>   
<p id="ref15">[15]Cho K, Van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[J]. arXiv preprint arXiv:1406.1078, 2014.</p>   
<p id="ref16">[16]Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.</p>   
<p id="ref17">[17]Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.</p>   
<p id="ref18">[18]Ross B, Rist M, Carbonell G, et al. Measuring the reliability of hate speech annotations: The case of the european refugee crisis[J]. arXiv preprint arXiv:1701.08118, 2017.</p>   
<p id="ref19">[19]Kolhatkar V, Wu H, Cavasso L, et al. The SFU opinion and comments corpus: A corpus for the analysis of online news comments[J]. Corpus pragmatics, 2020, 4: 155-190.</p>  
