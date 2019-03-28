##  1. Google News dataset （抽取）

### `句子——句子`

http://storage.googleapis.com/sentencecomp/compressiondata.json

### size：

200,000

### 论文：

《Sentence compression by deletion with lstms》

《Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains》

《Improving sentence compression by learning to predict gaze》

《Higher-order Syntactic Attention Network for Long Sentence Compression》

##	2. BBC News dataset（BROADCAST news）（抽取）

### `句子——句子`

Clarke和Lapata（2008）收集的大约1,500个句子对。 这些句子来自英国国家语料库（BNC）和2008年之前的美国新闻文本语料库

https://www.jamesclarke.net/research/resources

### size：

1,500

### 论文：

《Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains》

《Improving sentence compression by learning to predict gaze》

##	3. CNN / DailyMail （生成）

### `文章——多句子`

Hermann等人构建，2015

https://github.com/abisee/cnn-dailymail

### size：

Daily Mail语料库包含196,557 training，12,147 valid和10,396 test。

CNN/Daily Mail语料库包含286,722 training，13,362 valid和11,480 test。

训练集中每个文档大约有28个句子，摘要中平均有3-4个句子。训练集中每个文档的平均字数为802。

### 论文：

《SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents》

##	4. MSR Abstractive Text Compression Dataset（生成）

### `句子和短文本——句子`

https://www.microsoft.com/en-us/download/details.aspx?id=54262

来源：

Open American National Corpus (OANC)

http://www.anc.org/data/oanc

### size：

它包含6,169个源文本，包含多个压缩（共26,423对源和压缩文本），每个text最多五个压缩文本
在所有源文本中，3,769是单句，其余是2句短文本。 每对源和压缩文本由单语对齐器Jacana（Yao
等，2013）对齐。 数据集分为训练集（21,145对），验证集（1,908对）和测试集（3,370对）。

### 论文：

《An Operation Network for Abstractive Sentence Compression》

##	5. Gigaword（生成）

### `文章——标题`

Napoles et al., 2012

使用Rush（2015）提供的脚本预处理数据，这些脚本生成大约3.8M的训练样例和400K验证示例

https://github.com/facebookarchive/NAMAS

### 论文：

提取第一个句子：

Rush（2015）《A neural attention model for abstractive sentence summarization》

Fevry（2018）《Unsupervised Sentence Compression using Denoising Auto-Encoders》

提取前两个句子：

Nallapati （2016）《Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond》



