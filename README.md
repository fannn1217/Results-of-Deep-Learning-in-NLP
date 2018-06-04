# State-of-the-art-results-of-Deep-Learning

UPDATE:18-6-4

State of the art  results for  machine learning problems in NLP and CV, basically used deep learning methods.
Some state-of-the-art (SoTA) results, containing Paper, Datasets, Metric, Source Code, Year and Notes.

一些NLP & CV最新结果，包括论文，数据集，评估结果，源代码，年份以及阅读笔记。

---

# NLP

## Text Classification 文本分类
![image](https://github.com/fannn1217/Results-of-Deep-Learning-in-NLP-CV/blob/master/image/Text_Classification.png)

- `Accuracy` are the standard metrics.
- `Deep Learning`: deep learning features, deep learning method and RL.


|   Paper   | Yahoo | DBPedia | AGNews | Yelp P. | Yelp F. | ---- | Deep Learning |  RealTime  |
| :---------: | :----------: | :----------------: | :--------: | :--------------: | :-------: | :-------------: | :-----------: | :--------: |
|     LEAM     |  **77.42**   |        **99.02**        |    *92.45*     |  *95.31*  |     *64.09*     |      --       |       Y       |    --    |





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
      <td>LEAM：<a href='https://arxiv.org/pdf/1805.04174.pdf'>Joint Embedding of Words and Labels for Text Classification</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='https://github.com/guoyinwang/LEAM'>Tensorflow</a></ul></li></td>
      <td align="left">ACL</td> 
      <td align="left">2018</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1508.00200'>PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks</a></td>
      <td align="left"></td>
      <td align="left"></td>
      <td align="left"> <ul><li><a href='https://github.com/mnqu/PTE'>Code</a></ul></li></td>
      <td align="left">KDD</td> 
      <td align="left">2015</td>   
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

* **PTE** “PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks” **KDD(2015)**
  [[paper](https://arxiv.org/abs/1508.00200)]
  [[github](https://github.com/mnqu/PTE)]
  
目标是学习针对给定文本分类任务进行优化的文本表示。基本思想是在学习文本嵌入时将标记和未标记的信息结合起来。为了达到这个目的，首先需要有一个统一的表示来编码这两种信息。本文提出了实现这个目标的不同类型的网络，包括词 - 词共现网络，词 - 文档网络和词 - 标签网络，将三种网络结合起来，通过最小化经验分布函数（距离），学习到单词的向量表示，平均单词的向量表示即可获得文本表示，进而完成文本分类任务。

* “Multi-Task Label Embedding for Text Classification” **Arxiv（2017.10）**
  [[paper](https://arxiv.org/abs/1710.07210)]
  [[reading note](https://zhuanlan.zhihu.com/p/37669263)]
将文本分类中的标签转换为语义向量，从而将原始任务转换为向量匹配任务。实现了多任务标签嵌入的无监督，监督和半监督模型，所有这些模型都利用了任务之间的语义相关性。

