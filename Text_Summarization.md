# Text Summarization

**`Extraction (deletion-based)（word level）`**

* **LSTMs**  “Sentence Compression by Deletion with LSTMs” **EMNLP（2015）**
  [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf)]
  
  数据集：`Google News dataset`
  
序列标注的思路
提出了一种deletion-based LSTM方法，其任务是将句子转换为零和1的序列，对应于token删除决策。
  
* “Improving sentence compression by learning to predict gaze” **ACL（2016）**
  [[github](https://github.com/tatsuokun/sentence_compression)]

* “Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains” **ACL（2017）**
  [[paper](http://www.aclweb.org/anthology/P17-1127)]
  [[github](https://github.com/cnap/sentence-compression)]
  
  数据集：`Google News dataset、BBC News dataset（BROADCAST news）`

* “A Language Model based Evaluator for Sentence Compression” **ACL（2018）**
  [[paper](https://aclweb.org/anthology/P18-2028)]
  [[github](https://github.com/code4conference/code4sc)]
  [[reading note](https://zhuanlan.zhihu.com/p/50378570)]
  
  数据集：`Google News dataset 、Gigaword`
  
强化学习（无监督的方法）
基于删除的句子压缩旨在从源句中删除不必要的单词以形成短句，同时保证符合语法规范和遵循源句的基本含义。 以前的工作使用基于机器学习的方法或基于句法树的方法来产生最具可读性和信息量的压缩结果。然而使用RNN作为模型仍然会产生不合语法的句子，原因在于RNN的优化目标是基于单个词而不是整个压缩句子， 优化目标和评估之间存在差异。 因此，本文提出了以下两点改进：（i）将整个压缩句子的可读性作为学习目标;（ii）构建基于语言模型的评估器，用以恢复语法错误

* **HiSAN** “Higher-order Syntactic Attention Network for Long Sentence Compression” **NAACL（2018）**
[[reading note](https://zhuanlan.zhihu.com/p/53954265)]

  数据集：`Google News dataset`

在句子压缩任务中，神经网络方法已经成为主流研究方向，然而在面对长句子时效果不佳，本文提出了HiSAN，把higher-order dependency features作为attention。另外，为了弥补句法树的解析错误，将attention和最大化概率作为共同的训练目标。

***
  
**`Extraction (限定词典生成问题)（word level）`**

* **Ptr-Net** “Pointer Networks” **NIPS（2015）**
[[reading note](https://zhuanlan.zhihu.com/p/30860157)]

Pointer Networks预测的时候每一步都找当前输入序列中权重最大的那个元素，而由于输出序列完全来自输入序列，它可以适应输入序列的长度变化。不像attention mechanism将输入信息通过encoder整合成context vector，而是将attention转化为一个pointer，来选择原来输入序列中的元素。

* **NN-WE/NN-SE** “Neural Summarization by Extracting Sentences and Words” **ACL（2016）**
[[paper](https://www.aclweb.org/anthology/P16-1046)]

  数据集：`CNN / DailyMail（做了调整）`

本文针对的任务分为sentence和word两个level的summarization。sentence level是一个序列标签问题，每个句子有0或1两个标签。而word level则是一个限定词典规模下的生成问题，词典规模限定为原文档中所有出现的词。使用的模型也比较有特点，首先在encoder端将document分为word和sentence来encode，word使用CNN encode得到句子表示，接着将句子表示输入RNN得到encoder端隐藏层状态。从word到sentence的encode体现了本文的hierarchical document encoder的概念。

* “Unsupervised Sentence Compression using Denoising Auto-Encoders” **CoNLL（2018）**
  [[paper](https://arxiv.org/abs/1809.02669)]
  [[reading note](https://zhuanlan.zhihu.com/p/52521973)]
  
  数据集：`Gigaword`
  
为了弥补语料的缺乏问题，本文采用DAE作为策略，构建端到端的无监督模型，专注抽取式方法。

***

**`Extraction（sentence level）`**

* **SummaRuNNer** “SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents” **AAAI（2017）**
[[github](https://arxiv.org/abs/1611.04230)]

  数据集：`CNN / DailyMail（做了调整）`

提出了一个基于序列分类器的循环神经网络模型：SummaRuNNer，该模型表述简单，可解释性强，提出新的训练机制：使用生成式摘要(abstractive summary)的模式来训练抽取式任务

***

**`Abstraction`**

* **ABS** "A Neural Attention Model for Abstractive Sentence Summarization" **EMNLP (Rush, 2015)**
  [[paper](https://arxiv.org/abs/1509.00685)]
  [[github](https://github.com/facebookarchive/NAMAS)]
  
  数据集：`Gigaword`
  
考虑yc，yc是 yi+1的前C个词（不够就做padding），可以看做是yi+1的上下文信息（context）。
ABS+：与 extractive 的方法结合，就有了 ABS+ 模型。即在每次解码出一个词的时候，不仅考虑神经网络对当前词的预测概率 logp(yi+1|x,yc;θ)，还要开个窗口，去找一找当前窗口内的词是否在原文中出现过，如果有的话，概率会变大。

* “Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond” **CoNLL (Nallapati，2016)**

（1）在两种不同数据集上应用seq2seq+attention的模型，得到了state-of-the-art结果。
（2）根据摘要问题的特点提出了针对性的模型，结果更优。
	• LVT词汇表限制
	• 本文使用了一些额外的features，比如：词性，命名实体标签，单词的TF和IDF。将features融入到了word embedding上
  • Switching Generator/Pointer：模型中decoder带有一个开关，如果开关状态是打开generator，则生成一个单词；如果是关闭，decoder则生成一个原文单词位置的指针，然后拷贝到摘要中。
	• Hierarchical Encoder with Hieratchical Attention
（3）提出了一个包括多句子摘要的数据集和基准

* **Read-Again** "Efficient summarization with read-again and copy mechanism" **(Zeng, 2016)**
  [[paper](https://arxiv.org/pdf/1611.03382v1.pdf)]
  [[reading note](https://zhuanlan.zhihu.com/p/24887544)]
  
  数据集：`Gigaword`
  
Encoder-decoder模型已经广泛用于sequence to sequence任务，比如机器翻译、文本摘要等。作者提出它还存在一些缺点，比如Encoder侧在计算一个词的表示的时候只考虑了在其之前读到的词；还有，Decoder侧普遍用很大的词表来解决OOV（Out Of Vocabulary）的问题，从而导致解码缓慢。作者提出了对应的两个方法来解决这两个问题，一个就是Read-Again，即在产生词的表示之前预先“读”一遍句子，再就是作者提出“copy”机制，利用很小的词表来处理OOV问题，并且取得了state of art的效果。

* **Copy-net** "Incorporating Copying Mechanism in Sequence-to-Sequence Learning" **ACL（2016）**
[[reading note](https://zhuanlan.zhihu.com/p/48959800)]

  数据集：`LCSTS dataset（中文微博）`

和Pointer-Generator Networks很像。
模型包含两个部分：Generation-Mode用来根据词汇表生成词汇，然后Copy-Mode用来直接复制输入序列中的一些词。1.在词汇表上的概率分布，2.在输入序列上的概率分布，将这两部分的概率进行加和即得到最终的预测结果。

* **Pointer-Generator network** “Get To The Point: Summarization with Pointer-Generator Networks” **ACL（2017）**
[[reading note](https://zhuanlan.zhihu.com/p/27272224)]
[[github](https://github.com/becxer/pointer-generator/)]

  数据集：`CNN / DailyMail`

把sequence-to-sequence模型应用于生成摘要时存在两个主要的问题：（1）难以准确复述原文的事实细节、无法处理原文中的未登录词(OOV)；（2）生成的摘要中存在重复的片段。针对这两个问题，本文提出融合了seq2seq模型和pointer network的pointer-generator network以及覆盖率机制(coverage mechanism)
一方面通过seq2seq模型保持抽象生成的能力，另一方面通过pointer network直接从原文中取词，提高摘要的准确度和缓解OOV问题。
在预测的每一步，通过动态计算一个生成概率，把二者软性地结合起来
  
* **OperationNet** “An Operation Network for Abstractive Sentence Compression” **COLING（2018）**
[[reading note](https://zhuanlan.zhihu.com/p/58985964)]

  数据集：`MSR Abstractive Text Compression Dataset`

句子压缩会压缩句子，同时保留其最重要的内容。 基于删除的模型具有删除冗余单词的能力，而基于生成的模型能够对单词进行重新排序。 本文提出了operation network，一种用于抽象句子压缩的方法，它结合了基于删除和基于生成的句子压缩模型的优点。
在Pointer-Generator network的基础上，添加了delete decoder，对attention重新分布，同样是1.在词汇表上的概率分布，2.在输入序列上的概率分布，这两部分的概率加和得到最终的预测结果。

* “Generating topic-oriented summaries using neural attention” **NAACL（2018）**
[[reading note](https://zhuanlan.zhihu.com/p/60324533)]

  数据集：`CNN / DailyMail（做了调整）`

一篇文章可以涵盖几个topic，本文以生成针对不同主题的摘要为目标，将一篇文章与感兴趣的主题作为输入。由于缺少包含多个面向主题的文本摘要的数据集，本文从CNN / Dailymail数据集中人为构建语料。
模型采用Pointer-Generator network。将topic vector和input embedding concat起来作为输入句。



