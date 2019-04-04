## Text Summarization

**`Extraction (deletion-based)（word level）`**

* **LSTMs**  “Sentence Compression by Deletion with LSTMs” **EMNLP（2015）**
  [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf)]
  序列标注的思路
  提出了一种deletion-based LSTM方法，其任务是将句子转换为零和1的序列，对应于token删除决策。
  
* “Improving sentence compression by learning to predict gaze” **ACL（2016）**
  [[github](https://github.com/tatsuokun/sentence_compression)]

* “Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains” **ACL（2017）**
  [[paper](http://www.aclweb.org/anthology/P17-1127)]
  [[github](https://github.com/cnap/sentence-compression)]

* “A Language Model based Evaluator for Sentence Compression” **ACL（2018）**
  [[paper](https://aclweb.org/anthology/P18-2028)]
  [[github](https://github.com/code4conference/code4sc)]
  [[reading note](https://zhuanlan.zhihu.com/p/50378570)]
  
基于删除的句子压缩旨在从源句中删除不必要的单词以形成短句，同时保证符合语法规范和遵循源句的基本含义。 以前的工作使用基于机器学习的方法或基于句法树的方法来产生最具可读性和信息量的压缩结果。然而使用RNN作为模型仍然会产生不合语法的句子，原因在于RNN的优化目标是基于单个词而不是整个压缩句子， 优化目标和评估之间存在差异。 因此，本文提出了以下两点改进：（i）将整个压缩句子的可读性作为学习目标;（ii）构建基于语言模型的评估器，用以恢复语法错误
  
**`Extraction (限定词典生成问题)（word level）`**

* **Ptr-Net** “Pointer Networks” **NIPS（2015）**
[[reading note](https://zhuanlan.zhihu.com/p/30860157)]

Pointer Networks预测的时候每一步都找当前输入序列中权重最大的那个元素，而由于输出序列完全来自输入序列，它可以适应输入序列的长度变化。不像attention mechanism将输入信息通过encoder整合成context vector，而是将attention转化为一个pointer，来选择原来输入序列中的元素。

* **NN-WE/NN-SE** “Neural Summarization by Extracting Sentences and Words” **ACL（2016）**
[[paper](https://www.aclweb.org/anthology/P16-1046)]

本文针对的任务分为sentence和word两个level的summarization。sentence level是一个序列标签问题，每个句子有0或1两个标签。而word level则是一个限定词典规模下的生成问题，词典规模限定为原文档中所有出现的词。使用的模型也比较有特点，首先在encoder端将document分为word和sentence来encode，word使用CNN encode得到句子表示，接着将句子表示输入RNN得到encoder端隐藏层状态。从word到sentence的encode体现了本文的hierarchical document encoder的概念。

* “Unsupervised Sentence Compression using Denoising Auto-Encoders” **CoNLL（2018）**
  [[paper](https://arxiv.org/abs/1809.02669)]
  [[reading note](https://zhuanlan.zhihu.com/p/52521973)]
  
为了弥补语料的缺乏问题，本文采用DAE作为策略，构建端到端的无监督模型，专注抽取式方法。

**`Abstraction`**

* **ABS** "A Neural Attention Model for Abstractive Sentence Summarization" **EMNLP (Rush, 2015)**
  [[paper](https://arxiv.org/abs/1509.00685)]
  [[github](https://github.com/facebookarchive/NAMAS)]
  
考虑yc，yc是 yi+1的前C个词（不够就做padding），可以看做是yi+1的上下文信息（context）。
ABS+：与 extractive 的方法结合，就有了 ABS+ 模型。即在每次解码出一个词的时候，不仅考虑神经网络对当前词的预测概率 logp(yi+1|x,yc;θ)，还要开个窗口，去找一找当前窗口内的词是否在原文中出现过，如果有的话，概率会变大。

* **Read-Again** "Efficient summarization with read-again and copy mechanism" **(Zeng, 2016)**
  [[paper](https://arxiv.org/pdf/1611.03382v1.pdf)]
  [[reading note](https://zhuanlan.zhihu.com/p/24887544)]
  
Encoder-decoder模型已经广泛用于sequence to sequence任务，比如机器翻译、文本摘要等。作者提出它还存在一些缺点，比如Encoder侧在计算一个词的表示的时候只考虑了在其之前读到的词；还有，Decoder侧普遍用很大的词表来解决OOV（Out Of Vocabulary）的问题，从而导致解码缓慢。作者提出了对应的两个方法来解决这两个问题，一个就是Read-Again，即在产生词的表示之前预先“读”一遍句子，再就是作者提出“copy”机制，利用很小的词表来处理OOV问题，并且取得了state of art的效果。

* **Copy-net** "Incorporating Copying Mechanism in Sequence-to-Sequence Learning" **ACL（2016）**
[[reading note](https://zhuanlan.zhihu.com/p/48959800)]

和Pointer-Generator Networks很像。
模型包含两个部分：Generation-Mode用来根据词汇表生成词汇，然后Copy-Mode用来直接复制输入序列中的一些词。1.在词汇表上的概率分布，2.在输入序列上的概率分布，将这两部分的概率进行加和即得到最终的预测结果。

* **Pointer-Generator network** “Get To The Point: Summarization with Pointer-Generator Networks” **ACL（2017）**
[[reading note](https://zhuanlan.zhihu.com/p/27272224)]

把sequence-to-sequence模型应用于生成摘要时存在两个主要的问题：（1）难以准确复述原文的事实细节、无法处理原文中的未登录词(OOV)；（2）生成的摘要中存在重复的片段。针对这两个问题，本文提出融合了seq2seq模型和pointer network的pointer-generator network以及覆盖率机制(coverage mechanism)
一方面通过seq2seq模型保持抽象生成的能力，另一方面通过pointer network直接从原文中取词，提高摘要的准确度和缓解OOV问题。
在预测的每一步，通过动态计算一个生成概率，把二者软性地结合起来
  




