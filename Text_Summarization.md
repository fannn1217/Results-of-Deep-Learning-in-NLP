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

