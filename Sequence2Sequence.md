## Sequence2Sequence

### `Transformer`

* **Transformer**  “Attention is All You Need”  **NIPS(2017)**
  [[paper](https://arxiv.org/pdf/1706.03762.pdf)]
  [[pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)]
  [[keras](https://github.com/Lsdefine/attention-is-all-you-need-keras)]
  [[tensorflow](https://github.com/Kyubyong/transformer)]
  [[reading note](https://zhuanlan.zhihu.com/p/27469958)]
  
神经翻译的文章，抛弃了传统Encoder-Decoder中经典的卷积和循环结构，仅保留了attention，实现了并行计算。在Encoder层中，模型用了一个Multi-head self-attention以及一个全连接前馈网络，而在Decoder层中除了以上两部分之外又加入了一个对Encoder的attention层。从细节上来讲，attention层以及FCN都是残差链接并且在输出端进行了一次LayerNorm。
