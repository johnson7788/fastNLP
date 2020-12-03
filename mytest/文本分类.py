#!/usr/bin/env python
# coding: utf-8

# ## 文本分类(Text classification)
# 文本分类任务是将一句话或一段话划分到某个具体的类别。比如垃圾邮件识别，文本情绪分类等。
# 
# Example::   
# 1,商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!

# ## 步骤
# 一共有以下的几个步骤  
# (1) 读取数据  
# (2) 预处理数据  
# (3) 选择预训练词向量  
# (4) 创建模型  
# (5) 训练模型  

# (1) 读取数据
from fastNLP.io import ChnSentiCorpLoader

loader = ChnSentiCorpLoader()  # 初始化一个中文情感分类的loader
data_dir = loader.download()  # 这一行代码将自动下载数据到默认的缓存地址, 并将该地址返回
data_bundle = loader.load(data_dir)  # 这一行代码将从{data_dir}处读取数据至DataBundle

print(data_bundle)
# 可以看出，该data_bundle中一个含有三个\ref{DataSet}。通过下面的代码，我们可以查看DataSet的基本情况
print(data_bundle.get_dataset('train')[:2])  # 查看Train集前两个sample

# ### (2) 预处理数据
# 在NLP任务中，预处理一般包括: (a)将一整句话切分成汉字或者词; (b)将文本转换为index
from fastNLP.io import ChnSentiCorpPipe

pipe = ChnSentiCorpPipe()
data_bundle = pipe.process(data_bundle)  # 所有的Pipe都实现了process()方法，且输入输出都为DataBundle类型

print(data_bundle)  # 打印data_bundle，查看其变化

# 可以看到除了之前已经包含的3个\ref{DataSet}, 还新增了两个\ref{Vocabulary}。我们可以打印DataSet中的内容


print(data_bundle.get_dataset('train')[:2])

# 新增了一列为数字列表的chars，以及变为数字的target列。可以看出这两列的名称和刚好与data_bundle中两个Vocabulary的名称是一致的，我们可以打印一下Vocabulary看一下里面的内容。


char_vocab = data_bundle.get_vocab('chars')
print(char_vocab)

# Vocabulary是一个记录着词语与index之间映射关系的类，比如


index = char_vocab.to_index('选')
print("'选'的index是{}".format(index))  # 这个值与上面打印出来的第一个instance的chars的第一个index是一致的
print("index:{}对应的汉字是{}".format(index, char_vocab.to_word(index)))

# ### (3) 选择预训练词向量  
# 由于Word2vec, Glove, Elmo, Bert等预训练模型可以增强模型的性能，所以在训练具体任务前，选择合适的预训练词向量非常重要。在fastNLP中我们提供了多种Embedding使得加载这些预训练模型的过程变得更加便捷。更多关于Embedding的说明可以参考\ref{Embedding}。这里我们先给出一个使用word2vec的中文汉字预训练的示例，之后再给出一个使用Bert的文本分类。这里使用的预训练词向量为'cn-fastnlp-100d'，fastNLP将自动下载该embedding至本地缓存，fastNLP支持使用名字指定的Embedding以及相关说明可以参见\ref{Embedding}


from fastNLP.embeddings import StaticEmbedding

word2vec_embed = StaticEmbedding(char_vocab, model_dir_or_name='cn-char-fastnlp-100d')

# ### (4) 创建模型
# 这里我们使用到的模型结构如下所示，补图


from torch import nn
from fastNLP.modules import LSTM
import torch


# 定义模型
class BiLSTMMaxPoolCls(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=400, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = embed

        self.lstm = LSTM(self.embed.embedding_dim, hidden_size=hidden_size // 2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, chars, seq_len):  # 这里的名称必须和DataSet中相应的field对应，比如之前我们DataSet中有chars，这里就必须为chars
        # chars:[batch_size, max_len]
        # seq_len: [batch_size, ]
        chars = self.embed(chars)
        outputs, _ = self.lstm(chars, seq_len)
        outputs = self.dropout_layer(outputs)
        outputs, _ = torch.max(outputs, dim=1)
        outputs = self.fc(outputs)

        return {'pred': outputs}  # [batch_size,], 返回值必须是dict类型，且预测值的key建议设为pred


# 初始化模型
model = BiLSTMMaxPoolCls(word2vec_embed, len(data_bundle.get_vocab('target')))

# ### (5) 训练模型
# fastNLP提供了Trainer对象来组织训练过程，包括完成loss计算(所以在初始化Trainer的时候需要指定loss类型)，梯度更新(所以在初始化Trainer的时候需要提供优化器optimizer)以及在验证集上的性能验证(所以在初始化时需要提供一个Metric)


from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
metric = AccuracyMetric()
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=32, n_epochs=2, dev_data=data_bundle.get_dataset('dev'),
                  metrics=metric, device=device)
trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型

# 在测试集上测试一下模型的性能
from fastNLP import Tester

print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
tester.test()

# ### 使用Bert进行文本分类


# 只需要切换一下Embedding即可
from fastNLP.embeddings import BertEmbedding

# 这里为了演示一下效果，所以默认Bert不更新权重
bert_embed = BertEmbedding(char_vocab, model_dir_or_name='cn', auto_truncate=True, requires_grad=False)
model = BiLSTMMaxPoolCls(bert_embed, len(data_bundle.get_vocab('target')), )

import torch
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-5)
metric = AccuracyMetric()
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=16, dev_data=data_bundle.get_dataset('test'),
                  metrics=metric, device=device, n_epochs=3)
trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型

# 在测试集上测试一下模型的性能
from fastNLP import Tester

print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
tester.test()
