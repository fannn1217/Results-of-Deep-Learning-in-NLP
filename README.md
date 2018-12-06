# State-of-the-art-results-of-Deep-Learning

UPDATE:18-7-12

State of the art  results for  machine learning problems in NLP, basically used deep learning methods.
Some state-of-the-art (SoTA) results, containing Paper, Datasets, Metric, Source Code, Year and Notes.

一些NLP最新结果，包括论文，数据集，评估结果，源代码，年份以及阅读笔记。

---


## Text Classification 

[[综述参考-用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践](https://zhuanlan.zhihu.com/p/25928551)]

![image](https://github.com/fannn1217/Results-of-Deep-Learning-in-NLP-CV/blob/master/image/Text_Classification.png)

- `Accuracy` are the standard metrics.
- `Deep Learning`: deep learning features, deep learning method and RL.


|   Paper   | Yahoo | DBPedia | AGNews | Yelp P. | Yelp F. | Amazon P. | Amazon F. | Deep Learning | 
| :---------: | :----------: | :----------: | :--------: | :-----------: | :-------: | :-----------: | :-----------: | :--------: |
|     LEAM     |  *77.42*   |        *99.02*        |    *92.45*     |  *95.31*  |     *64.09*     |      --       |      --       |       Y       | 
|     Region.Emb.     |        --       |      *98.9*        |    *92.8*     |  *96.4*  |     *64.9*     |      *95.1*       |      *60.9*       |       Y       |  
|     Dense.CNN     |      --     |        **99.2**        |    **93.6**     |  **96.5**  |     **66.0**     |      --       |      **63.0**       |       Y       |


`待看：`
<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="10%">Datasets</th>
      <th align="center" width="10%">Metric</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Published</th>
      <th align="center" width="10%">Year</th>
      <th align="center" width="20%">Reading Note</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1705.09207'> Learning Structured Text Representations </a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 68.6</td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></ul></li></td>
      <td align="left">TACL</td> 
      <td align="left">2018</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1710.00519'>Attentive Convolution</a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 67.36</td>
      <td align="left"> <ul><li><a href='https://github.com/yinwenpeng/Attentive_Convolution'>Theano</a></ul></li></td>
      <td align="left">arxiv</td> 
      <td align="left">2017.10</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1704.05742'>Adversarial Multi-task Learning for Text Classification</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='http://pfliu.com/paper/adv-mtl.html'>Theano</a></ul></li></td>
      <td align="left">ACL</td> 
      <td align="left">2017</td> 
      <td align="left"><a href='https://blog.csdn.net/qj8380078/article/details/79914170'>CSDN</a></td>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1710.10393'>
Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='https://github.com/lancopku/label-embedding-network'>Pytorch+Tensorflow</a></ul></li></td>
      <td align="left">arxiv</td> 
      <td align="left">2017.10</td> 
      <td align="left"><a href=''></a></td>
    </tr>
    <tr>
      <td><a href='http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf'>
Large-Scale Hierarchical Text Classification with Recursively Regularized Deep Graph-CNN</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='https://github.com/HKUST-KnowComp/DeepGraphCNNforTexts'>Tensorflow</a></ul></li></td>
      <td align="left">WWW</td> 
      <td align="left">2018</td> 
      <td align="left"><a href=''></a></td>
    </tr>
  </tbody>
</table>

`已看`

### `Embedding`

* **PTE** “PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks” **KDD(2015)**
  [[paper](https://arxiv.org/abs/1508.00200)]
  [[github](https://github.com/mnqu/PTE)]
  
目标是学习针对给定文本分类任务进行优化的文本表示。基本思想是在学习文本嵌入时将标记和未标记的信息结合起来。为了达到这个目的，首先需要有一个统一的表示来编码这两种信息。本文提出了实现这个目标的不同类型的网络，包括词 - 词共现网络，词 - 文档网络和词 - 标签网络，将三种网络结合起来，通过最小化经验分布函数（距离），学习到单词的向量表示，平均单词的向量表示即可获得文本表示，进而完成文本分类任务。

* **LEAM** “Joint Embedding of Words and Labels for Text Classification” **ACL(2018)**
  [[paper](https://arxiv.org/pdf/1805.04174.pdf)]
  [[github(Tensorflow)](https://github.com/guoyinwang/LEAM)]
  
将文本分类视为标签 - 词联合嵌入问题：每个标签与词向量嵌入在同一空间中。贡献：利用attention机制 联合利用标签embedding 

* **Region.Emb.**  "A New Method Of Region Embedding For Text Classification" **ICLR(2018)**
  [[paper](https://openreview.net/pdf?id=BkSDMA36Z)]
  [[github(Pytorch)](https://github.com/schelotto/Region_Embedding_Text_Classification_Pytorch)]
  [[github(Tensorflow)](https://github.com/text-representation/local-context-unit)]
  [[reading note](https://zhuanlan.zhihu.com/p/39264740)]
  
针对文本分类，本文提出了一种n-gram新的分布式表示——region embedding。在模型中，单词由两部分表示 1单词本身的embedding 2 联系上下文的权重矩阵。

### `多任务`

* **MTLE** “Multi-Task Label Embedding for Text Classification” **Arxiv（2017.10）**
  [[paper](https://arxiv.org/abs/1710.07210)]
  [[reading note](https://zhuanlan.zhihu.com/p/37669263)]
  
将文本分类中的标签转换为语义向量，从而将原始任务转换为向量匹配任务。实现了多任务标签嵌入的无监督，监督和半监督模型，所有这些模型都利用了任务之间的语义相关性。

*  “Recurrent Neural Network for Text Classification with Multi-Task Learning” **IJCAI（2016）**
  [[paper](http://www.ijcai.org/Proceedings/16/Papers/408.pdf)]
  [[reading note](https://zhuanlan.zhihu.com/p/27562717?refer=xitucheng10)]
  
针对文本多分类任务，提出了基于RNN的三种不同的共享信息机制对具有特定任务和文本进行建模，在四个基准的文本分类任务中取得了较好的结果。
三种model：Uniform-Layer Architecture、Coupled-Layer Architecture、Shared-Layer Architecture

*  “A Generalized Recurrent Neural Architecturefor Text Classification with Multi-Task Learning” **IJCAI（2017）**
  [[paper](http://www.ijcai.org/proceedings/2017/0473.pdf)]
  
上一篇文章的后续工作，提出了一个通用的架构，将三种layer结合起来。每个任务都拥有一个基于LSTM的single layer，用于任务内学习。Pair-wise Coupling layer和local fusion layer设计用于直接和间接的任务间交互。利用global fusion layer来维护全局内存，以便在所有任务之间共享信息

### `CNN`

* **Dense.CNN** “Densely Connected CNN with Multi-scale Feature Attention for Text Classification” **IJCAI(2018)**
  [[paper](http://coai.cs.tsinghua.edu.cn/hml/media/files/2018wangshiyao_DenselyCNN.pdf)]
  [[github](https://github.com/wangshy31/Densely-Connected-CNN-with-Multiscale-Feature-Attention)]
  [[reading note](https://zhuanlan.zhihu.com/p/39704684)]
  
传统的CNN有固定的卷积核大小，无法在CNN模型中自适应地选择多尺度特征进行文本分类。收到DenseNet启发，本文提出了一种新的CNN模型，该模型在卷积层之间具有密集连接，并具有多尺度特征注意机制。通过这两个设计考虑因素，该模型能够自适应地选择用于文本分类的多尺度特征。

---

## Sequence2Sequence

### `Transformer`

* **Transformer**  “Attention is All You Need”  **NIPS(2017)**
  [[paper](https://arxiv.org/pdf/1706.03762.pdf)]
  [[pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)]
  [[keras](https://github.com/Lsdefine/attention-is-all-you-need-keras)]
  [[tensorflow](https://github.com/Kyubyong/transformer)]
  [[reading note](https://zhuanlan.zhihu.com/p/27469958)]
  
神经翻译的文章，抛弃了传统Encoder-Decoder中经典的卷积和循环结构，仅保留了attention，实现了并行计算。在Encoder层中，模型用了一个Multi-head self-attention以及一个全连接前馈网络，而在Decoder层中除了以上两部分之外又加入了一个对Encoder的attention层。从细节上来讲，attention层以及FCN都是残差链接并且在输出端进行了一次LayerNorm。

---

## Pre-train

* **ULMFiT** “Universal Language Model Fine-tuning for Text Classification” **ACL(2018)**
  [[paper](https://arxiv.org/abs/1801.06146)]
  [[code](http://nlp.fast.ai/ulmfit)]
  [[reading note](https://www.jianshu.com/p/5b680f4fb2f2)]
  
  
* **GPT**  “Improving Language Understanding by Generative Pre-Training” **（2018）**
  [[paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)]
  [[github](https://github.com/openai/finetune-transformer-lm)]
  [[guide](https://finetune.indico.io/#)]
  
标注数据稀少，对无标记的数据进行pre-train是一种提高模型效果的方法。本文使用transformer（decoder only）在无标记语料上进行预训练language model，之后经过模型的fine-tuning，实现对QA，语义相似度，分类等任务的效果提升。
  
* **BERT**  “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” **（2018）**
  [[paper](https://arxiv.org/abs/1810.04805)]
  [[github](https://github.com/google-research/language)]
  [[reading note](https://www.jiqizhixin.com/articles/2018-10-12-13)]
  
不同于GPT等方法（预训练语言模型后微调），本文使用多层双向 Transformer 编码器，不使用传统的从左到右或从右到左的语言模型来预训练 BERT，而是使用两个新型无监督预测任务来进行预训练。这两个新任务分别是Masked LM 和下一句预测。实验表明，BERT 刷新了 11 项 NLP 任务的当前最优性能记录。

---

## Sentence Compression

* **LSTMs**  “Sentence Compression by Deletion with LSTMs” **EMNLP（2015）**
  [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf)]


*   “A Language Model based Evaluator for Sentence Compression” **ACL（2018）**
  [[paper](https://aclweb.org/anthology/P18-2028)]
  [[github](https://github.com/code4conference/code4sc)]
  [[reading note](https://zhuanlan.zhihu.com/p/50378570)]
  
基于删除的句子压缩旨在从源句中删除不必要的单词以形成短句，同时保证符合语法规范和遵循源句的基本含义。 以前的工作使用基于机器学习的方法或基于句法树的方法来产生最具可读性和信息量的压缩结果。然而使用RNN作为模型仍然会产生不合语法的句子，原因在于RNN的优化目标是基于单个词而不是整个压缩句子， 优化目标和评估之间存在差异。 因此，本文提出了以下两点改进：（i）将整个压缩句子的可读性作为学习目标;（ii）构建基于语言模型的评估器，用以恢复语法错误
  
  
  
