# State-of-the-art-results-of-Deep-Learning

UPDATE:18-7-12

State of the art  results for  machine learning problems in NLP and CV, basically used deep learning methods.
Some state-of-the-art (SoTA) results, containing Paper, Datasets, Metric, Source Code, Year and Notes.

一些NLP & CV最新结果，包括论文，数据集，评估结果，源代码，年份以及阅读笔记。

---

# NLP

## Text Classification 文本分类

[[综述参考-用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践](https://zhuanlan.zhihu.com/p/25928551)]

![image](https://github.com/fannn1217/Results-of-Deep-Learning-in-NLP-CV/blob/master/image/Text_Classification.png)

- `Accuracy` are the standard metrics.
- `Deep Learning`: deep learning features, deep learning method and RL.


|   Paper   | Yahoo | DBPedia | AGNews | Yelp P. | Yelp F. | Amazon P. | Amazon F. | Deep Learning |  RealTime  |
| :---------: | :----------: | :----------: | :--------: | :-----------: | :-------: | :-----------: | :-----------: | :--------: | :--------: |
|     LEAM     |  *77.42*   |        **99.02**        |    *92.45*     |  *95.31*  |     *64.09*     |      --       |      --       |       Y       |    --    |
|     Region.Emb.     |        --       |      *98.9*        |    **92.8**     |  **96.4**  |     **64.9**     |      *95.1*       |      *60.9*       |       Y       |    --    |




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
      <td><a href='http://coai.cs.tsinghua.edu.cn/hml/media/files/2018wangshiyao_DenselyCNN.pdf'>Densely Connected CNN with Multi-scale Feature Attention for Text Classification</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='https://github.com/wangshy31/Densely-Connected-CNN-with-Multiscale-Feature-Attention'>NOT finish</a></ul></li></td>
      <td align="left">IJCAI</td> 
      <td align="left">2018</td>   
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
  </tbody>
</table>

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
