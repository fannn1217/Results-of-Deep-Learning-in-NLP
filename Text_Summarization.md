## Text Summarization

**`Sentence Compression`**

* **LSTMs**  “Sentence Compression by Deletion with LSTMs” **EMNLP（2015）**
  [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf)]

*  “Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains” **ACL（2017）**
  [[paper](http://www.aclweb.org/anthology/P17-1127)]
  [[github](https://github.com/cnap/sentence-compression)]

*   “A Language Model based Evaluator for Sentence Compression” **ACL（2018）**
  [[paper](https://aclweb.org/anthology/P18-2028)]
  [[github](https://github.com/code4conference/code4sc)]
  [[reading note](https://zhuanlan.zhihu.com/p/50378570)]
  
基于删除的句子压缩旨在从源句中删除不必要的单词以形成短句，同时保证符合语法规范和遵循源句的基本含义。 以前的工作使用基于机器学习的方法或基于句法树的方法来产生最具可读性和信息量的压缩结果。然而使用RNN作为模型仍然会产生不合语法的句子，原因在于RNN的优化目标是基于单个词而不是整个压缩句子， 优化目标和评估之间存在差异。 因此，本文提出了以下两点改进：（i）将整个压缩句子的可读性作为学习目标;（ii）构建基于语言模型的评估器，用以恢复语法错误
  
*   “Unsupervised Sentence Compression using Denoising Auto-Encoders” **CoNLL（2018）**
  [[paper](https://arxiv.org/abs/1809.02669)]
  [[reading note](https://zhuanlan.zhihu.com/p/52521973)]
  
为了弥补语料的缺乏问题，本文采用DAE作为策略，构建端到端的无监督模型，专注抽取式方法。

**`Document Summarization`**

* **ABS** "A Neural Attention Model for Abstractive Sentence Summarization" **EMNLP (Rush, 2015)**
  [[paper](https://arxiv.org/abs/1509.00685)]
  [[github](https://github.com/facebookarchive/NAMAS)]

* **Read-Again** "Efficient summarization with read-again and copy mechanism" **(Zeng, 2016)**
  [[paper](https://arxiv.org/pdf/1611.03382v1.pdf)]
  [[note](https://zhuanlan.zhihu.com/p/24887544)]
  
Encoder-decoder模型已经广泛用于sequence to sequence任务，比如机器翻译、文本摘要等。作者提出它还存在一些缺点，比如Encoder侧在计算一个词的表示的时候只考虑了在其之前读到的词；还有，Decoder侧普遍用很大的词表来解决OOV（Out Of Vocabulary）的问题，从而导致解码缓慢。作者提出了对应的两个方法来解决这两个问题，一个就是Read-Again，即在产生词的表示之前预先“读”一遍句子，再就是作者提出“copy”机制，利用很小的词表来处理OOV问题，并且取得了state of art的效果。


  
