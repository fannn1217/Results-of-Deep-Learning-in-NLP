# Text Summarization

**`Extraction (deletion-based)（word level）`**

* **LSTMs**  “Sentence Compression by Deletion with LSTMs” **EMNLP（2015）**
  [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf)]
  
  数据集：`Google News dataset`

```
序列标注的思路
提出了一种deletion-based LSTM方法，其任务是将句子转换为零和1的序列，对应于token删除决策。
```
  
* “Improving sentence compression by learning to predict gaze” **ACL（2016）**
  [[github](https://github.com/tatsuokun/sentence_compression)]
  
  数据集：`Google News dataset`

```
将eye-tracking用到压缩任务中，gaze predict和CCG标记预测是辅助训练任务
```

* “Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains” **ACL（2017）**
  [[paper](http://www.aclweb.org/anthology/P17-1127)]
  [[github](https://github.com/cnap/sentence-compression)]
  
  数据集：`Google News dataset、BBC News dataset（BROADCAST news）`

* “A Language Model based Evaluator for Sentence Compression” **ACL（2018）**
  [[paper](https://aclweb.org/anthology/P18-2028)]
  [[github](https://github.com/code4conference/code4sc)]
  [[reading note](https://zhuanlan.zhihu.com/p/50378570)]
  
  数据集：`Google News dataset 、Gigaword`

```
强化学习（无监督的方法）
基于删除的句子压缩旨在从源句中删除不必要的单词以形成短句，同时保证符合语法规范和遵循源句的基本含义。 以前的工作使用基于机器学习的方法或基于句法树的方法来产生最具可读性和信息量的压缩结果。然而使用RNN作为模型仍然会产生不合语法的句子，原因在于RNN的优化目标是基于单个词而不是整个压缩句子， 优化目标和评估之间存在差异。 因此，本文提出了以下两点改进：（i）将整个压缩句子的可读性作为学习目标;（ii）构建基于语言模型的评估器，用以恢复语法错误
```

* **HiSAN** “Higher-order Syntactic Attention Network for Long Sentence Compression” **NAACL（2018）**
[[reading note](https://zhuanlan.zhihu.com/p/53954265)]

  数据集：`Google News dataset`

```
在句子压缩任务中，神经网络方法已经成为主流研究方向，然而在面对长句子时效果不佳，本文提出了HiSAN，把higher-order dependency features作为attention。另外，为了弥补句法树的解析错误，将attention和最大化概率作为共同的训练目标。
```

***
  
**`Extraction (限定词典生成问题)（word level）`**

* **Ptr-Net** “Pointer Networks” **NIPS（2015）**
[[reading note](https://zhuanlan.zhihu.com/p/30860157)]

```
Pointer Networks预测的时候每一步都找当前输入序列中权重最大的那个元素，而由于输出序列完全来自输入序列，它可以适应输入序列的长度变化。不像attention mechanism将输入信息通过encoder整合成context vector，而是将attention转化为一个pointer，来选择原来输入序列中的元素。
```

* **NN-WE/NN-SE** “Neural Summarization by Extracting Sentences and Words” **ACL（2016）**
[[paper](https://www.aclweb.org/anthology/P16-1046)]

  数据集：`CNN / DailyMail（做了调整）`

```
本文针对的任务分为sentence和word两个level的summarization。sentence level是一个序列标签问题，每个句子有0或1两个标签。而word level则是一个限定词典规模下的生成问题，词典规模限定为原文档中所有出现的词。使用的模型也比较有特点，首先在encoder端将document分为word和sentence来encode，word使用CNN encode得到句子表示，接着将句子表示输入RNN得到encoder端隐藏层状态。从word到sentence的encode体现了本文的hierarchical document encoder的概念。
```

* “Unsupervised Sentence Compression using Denoising Auto-Encoders” **CoNLL（2018）**
  [[paper](https://arxiv.org/abs/1809.02669)]
  [[reading note](https://zhuanlan.zhihu.com/p/52521973)]
  
  数据集：`Gigaword`
  
```
为了弥补语料的缺乏问题，本文采用DAE作为策略，构建端到端的无监督模型，专注抽取式方法。
```

* **JECS** “Neural Extractive Text Summarization with Syntactic Compression” **EMNLP（2019）**
  [[github](https://github.com/jiacheng-xu/neu-compression-sum)]
  
  数据集：`CNN / DailyMail、NYT`

```
Model：
step1. 编码句子、文档（lstm+cnn）进行句子级别的extraction，类似pointer network的方式
step2. 对extract的句子进行句子压缩。根据规则和解析结果得到每个句子中可压缩的短语，句子压缩模块评估可压缩的选项，并决定是否删除。
       句子压缩：ELMO作为encoder，对选项二分类决定是否删除。
step3. 后处理：删除在其他地方出现过的完全一致的压缩短语

Training：
数据集：构建Oracle Label

Analysis：
1. human evaluation
2. 手动分析了50个例子，总结错误类型
3. 分类阈值（DEL> 0.5的概率）可能不是最优的，分析不同阈值的表现，实验中0.45最优
4. 不同句法成分的压缩结果分析
```

***

**`Extraction（sentence level）`**

* **SummaRuNNer** “SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents” **AAAI（2017）**
  [[paper](https://arxiv.org/abs/1611.04230)]
  [[reading note](https://zhuanlan.zhihu.com/p/51476934)]

  数据集：`CNN / DailyMail（做了调整）`

```
提出了一个基于序列分类器的循环神经网络模型：SummaRuNNer，该模型表述简单，可解释性强，提出新的训练机制：使用生成式摘要(abstractive summary)的模式来训练抽取式任务
```

* **RNES** “Learning to Extract Coherent Summary via Deep Reinforcement Learning” **AAAI（2018）**

  数据集：`CNN / DailyMail`

```
RNES：强化学习

为了提取连贯的摘要，我们提出了一种神经连贯模型来捕捉跨句子的语义和句法连贯模式。计算两个句子的相干性，将其添加进强化学习的reward中，使输出摘要保持连贯性。
```

* “Neural Latent Extractive Document Summarization” **EMNLP（2018）**
  [[paper](https://arxiv.org/abs/1808.07187)]
  [[reading note](https://blog.csdn.net/qq_30219017/article/details/87926142)]
  
  数据集：`CNN / DailyMail`

```
LSTM + 强化学习

过去的抽取式摘要技术多是把这个看作句子序列标注二分类问题或者句子排序问题。

这篇论文把句子对应的label视为句子的隐变量。不是最大化每个句子到训练数据(gold label)的可能性，而是最大化生成摘要是这个人工摘要整体的可能性。
```

* **SWAP-NET** “Extractive Summarization with SWAP-NET: Sentences and Words from Alternating Pointer Networks” **ACL（2018）**
  [[paper](https://www.aclweb.org/anthology/P18-1014/)]

  数据集：`CNN / DailyMail`（使用此数据集的匿名版本，来自Cheng和Lapata（2016），其中包含用于训练的重要句子标签；使用RAKE（一种无监督的关键词提取方法）从每个gold摘要中提取关键词）

```
SWAP-NET使用一个新的基于两级pointer-network的架构来模拟关键词和关键句子的交互。 

SWAP-NET识别输入文档中的关键句子和关键词，然后将它们组合起来形成提取摘要。
```

* **NEUSUM** “Neural Document Summarization by Jointly Learning to Score and Select Sentences” **ACL（2018）**
  [[github](https://res.qyzhou.me)]

  数据集：`CNN / DailyMail`

```
联合学习评分和选择句子，首先使用分层编码器读取文档句子，然后通过逐句提取句子来构建摘要。
根据之前时间步抽取的句子，对剩余句子评分，确定抽取的下一个句子。每抽取一个句子，就把此句子从呆抽取的集合中去掉。
优化目标：KL散度
```

* **HIBERT** “HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization” **ACL（2019）**

  数据集：`CNN / DailyMail`

```
两层transformer进行文档编码，并在使用标记数据预训练encoder（完形填空任务cloze）

实验分为三阶段：
1. 开放域预训练（mask）（encoder+decoder）
2. 目标数据集预训练（mask）（encoder+decoder）
3. 目标数据集fine-tune（摘要）（encoder+序列标注）
```

***

**`Abstraction`**

* **ABS** "A Neural Attention Model for Abstractive Sentence Summarization" **EMNLP (Rush, 2015)**
  [[paper](https://arxiv.org/abs/1509.00685)]
  [[github](https://github.com/facebookarchive/NAMAS)]
  
  数据集：`Gigaword`
  
```
考虑yc，yc是 yi+1的前C个词（不够就做padding），可以看做是yi+1的上下文信息（context）。
ABS+：与 extractive 的方法结合，就有了 ABS+ 模型。即在每次解码出一个词的时候，不仅考虑神经网络对当前词的预测概率 logp(yi+1|x,yc;θ)，还要开个窗口，去找一找当前窗口内的词是否在原文中出现过，如果有的话，概率会变大。
```

* “Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond” **CoNLL (Nallapati，2016)**
[[paper](https://arxiv.org/abs/1602.06023)]

  数据集：`Gigaword、CNN / DailyMail`

```
（1）在两种不同数据集上应用seq2seq+attention的模型，得到了state-of-the-art结果。

（2）根据摘要问题的特点提出了针对性的模型，结果更优。

    • LVT词汇表限制

    • 本文使用了一些额外的features，比如：词性，命名实体标签，单词的TF和IDF。将features融入到了word embedding上

    • Switching Generator/Pointer：模型中decoder带有一个开关，如果开关状态是打开generator，则生成一个单词；如果是关闭，decoder则生成一个原文单词位置的指针，然后拷贝到摘要中。

    • Hierarchical Encoder with Hieratchical Attention

（3）提出了一个包括多句子摘要的数据集和基准
```

* **ASC** “Language as a Latent Variable: Discrete Generative Models for Sentence Compression” **EMNLP（2016）**

  数据集：`Gigaword`

```
结合pointer network，提出了auto-encoding sentence compression (ASC)，无监督模型，和supervised forced-attention sentence compression (FSC)，监督模型，将二者组合实现了半监督学习。
```

* **Read-Again** "Efficient summarization with read-again and copy mechanism" **(Zeng, 2016)**
  [[paper](https://arxiv.org/pdf/1611.03382v1.pdf)]
  [[reading note](https://zhuanlan.zhihu.com/p/24887544)]
  
  数据集：`Gigaword`

```
Encoder-decoder模型已经广泛用于sequence to sequence任务，比如机器翻译、文本摘要等。作者提出它还存在一些缺点，比如Encoder侧在计算一个词的表示的时候只考虑了在其之前读到的词；还有，Decoder侧普遍用很大的词表来解决OOV（Out Of Vocabulary）的问题，从而导致解码缓慢。作者提出了对应的两个方法来解决这两个问题，一个就是Read-Again，即在产生词的表示之前预先“读”一遍句子，再就是作者提出“copy”机制，利用很小的词表来处理OOV问题，并且取得了state of art的效果。
```

* **Copy-net** "Incorporating Copying Mechanism in Sequence-to-Sequence Learning" **ACL（2016）**
  [[reading note](https://zhuanlan.zhihu.com/p/48959800)]

  数据集：`LCSTS dataset（中文微博）`

```
和Pointer-Generator Networks很像。不同的是生成概率和复制概率在这里直接加和。
模型包含两个部分：Generation-Mode用来根据词汇表生成词汇，然后Copy-Mode用来直接复制输入序列中的一些词。1.在词汇表上的概率分布，2.在输入序列上的概率分布，将这两部分的概率进行加和即得到最终的预测结果。
```

* **Pointer-Generator network** “Get To The Point: Summarization with Pointer-Generator Networks” **ACL（2017）**
  [[reading note](https://zhuanlan.zhihu.com/p/27272224)]
  [[github](https://github.com/becxer/pointer-generator/)]

  数据集：`CNN / DailyMail`
  
```
把sequence-to-sequence模型应用于生成摘要时存在两个主要的问题：（1）难以准确复述原文的事实细节、无法处理原文中的未登录词(OOV)；（2）生成的摘要中存在重复的片段。针对这两个问题，本文提出融合了seq2seq模型和pointer network的pointer-generator network以及覆盖率机制(coverage mechanism)
一方面通过seq2seq模型保持抽象生成的能力，另一方面通过pointer network直接从原文中取词，提高摘要的准确度和缓解OOV问题。
在预测的每一步，通过动态计算一个生成概率，把二者软性地结合起来
```

* **OperationNet** “An Operation Network for Abstractive Sentence Compression” **COLING（2018）**
  [[reading note](https://zhuanlan.zhihu.com/p/58985964)]

  数据集：`MSR Abstractive Text Compression Dataset`

```
句子压缩会压缩句子，同时保留其最重要的内容。 基于删除的模型具有删除冗余单词的能力，而基于生成的模型能够对单词进行重新排序。 本文提出了operation network，一种用于抽象句子压缩的方法，它结合了基于删除和基于生成的句子压缩模型的优点。
在Pointer-Generator network的基础上，添加了delete decoder，对attention重新分布，同样是1.在词汇表上的概率分布，2.在输入序列上的概率分布，这两部分的概率加和得到最终的预测结果。
```


* “A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss” **ACL（2018）**
  [[code](https://github.com/HsuWanTing/unified-summarization)]

  数据集：`CNN / DailyMail`

```
将abstracter和extractor（sentence level）做结合
extractor模型参照SummaRuNNer，得到句子级别的注意力分布
abstracter模型参照Pointer-Generator network，得到词级注意力分布
1. 句子级别的注意力用于调整词级关注度，使得较少出现的句子中的词语不太可能被生成。 
2. 引入了一种新颖的不一致性损失函数loss来约束两个级别的注意力之间的不一致性。sentence-level和word-level attention的共同学习：当词级注意力很高时，我们希望句子级别的注意力也很高
```


* “Generating topic-oriented summaries using neural attention” **NAACL（2018）**
  [[reading note](https://zhuanlan.zhihu.com/p/60324533)]

  数据集：`CNN / DailyMail（做了调整）`

```
一篇文章可以涵盖几个topic，本文以生成针对不同主题的摘要为目标，将一篇文章与感兴趣的主题作为输入。由于缺少包含多个面向主题的文本摘要的数据集，本文从CNN / Dailymail数据集中人为构建语料。
模型采用Pointer-Generator network。将topic vector和input embedding concat起来作为输入句。
```

* “Abstractive Summarization Using Attentive Neural Techniques” **ICON（2018）**
  
  数据集：`Gigaword`
  
```
1、使用了self-attention模型（transformer）

2、提出了新的evaluation方法（vert）
```


*  “Data-efficient Neural Text Compression with Interactive Learning” **NAACL（2019）**
  [[reading note](https://zhuanlan.zhihu.com/p/68391870)]
  
   数据集：`Google News dataset`
  
```
本文提出了一种新颖的交互式设置，通过采用主动学习，将模型迁移到新的领域，减少人为监督。

本文采用主动学习（AL）策略来：

（a）学习使用最小数据量的模型

（b）将使用小数据集的预训练模型迁移到新领域上
```

***

**`Unified Abstraction and Extraction`**


* “A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss” **ACL（2018）**
  [[code](https://github.com/HsuWanTing/unified-summarization)]

  数据集：`CNN / DailyMail`

```
将abstracter和extractor（sentence level）做结合
extractor模型参照SummaRuNNer，得到句子级别的注意力分布
abstracter模型参照Pointer-Generator network，得到词级注意力分布
1. 句子级别的注意力用于调整词级关注度，使得较少出现的句子中的词语不太可能被生成。 
2. 引入了一种新颖的不一致性损失函数loss来约束两个级别的注意力之间的不一致性。sentence-level和word-level attention的共同学习：当词级注意力很高时，我们希望句子级别的注意力也很高
两种训练方式：
首先均预训练extractor和abstracter
方式1. Two-stages training。抽取模型变成分类器以选择具有高关注的句子（即，句子attention>阈值）。 再简单地将抽取的句子提供给生成模型。
方式2. End-to-end training。句子级注意力β与词级注意力αtt结合。 通过最小化四个损失函数端到端训练抽取和生成模型：L(ext)，L(abs)，L(cov)，以及方程式中的L(inc)。
```


* **BERTSUM** “Text Summarization with Pretrained Encoders” **EMNLP（2019）**
  [[code](https://github.com/nlpyang/PreSumm)]

  数据集：`CNN / DailyMail、NYT、XSum`

```
本文展示了BERT在文本摘要中的应用。
1. 提取模型：建立在bert编码器之上，通过堆叠多个句子间transformer层。
2. 生成模型：encoder：bert，decoder：Transformer。同时提出了一种新的微调时间表，它对编码器和解码器采用不同的优化器（不同lr），作为避免两者之间不匹配的手段（前者是预训练而后者不是）。
3. Two-stage的微调方法，先在抽取微调，后在生成微调
```
