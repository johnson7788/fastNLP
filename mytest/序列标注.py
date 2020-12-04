#!/usr/bin/env python
# coding: utf-8

# # 序列标注
# ## 命名实体识别(name entity recognition, NER)
# 命名实体识别任务是从文本中抽取出具有特殊意义或者指代性非常强的实体，通常包括人名、地名、机构名和时间等。 如下面的例子中
# *我来自复旦大学*
# 其中“复旦大学”就是一个机构名，命名实体识别就是要从中识别出“复旦大学”这四个字是一个整体，且属于机构名这个类别。这个问题在实际做的时候会被 转换为序列标注问题
# 针对"我来自复旦大学"这句话，我们的预测目标将是[O, O, O, B-ORG, I-ORG, I-ORG, I-ORG]，其中O表示out,即不是一个实体，B-ORG是ORG( organization的缩写)这个类别的开头(Begin)，I-ORG是ORG类别的中间(Inside)。
# 

# ## 载入数据, 加载微博数据
from fastNLP.io import WeiboNERPipe

data_bundle = WeiboNERPipe().process_from_file()
print(data_bundle.get_dataset('train')[:2])

# ## 模型构建
# 
# 首先选择需要使用的Embedding类型。关于Embedding的相关说明可以参见《使用Embedding模块将文本转成向量》。 在这里我们使用通过word2vec预训练的中文汉字embedding。

# In[2]:


from fastNLP.embeddings import StaticEmbedding

embed = StaticEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name='cn-char-fastnlp-100d')

# 选择好Embedding之后，我们可以使用fastNLP中自带的 fastNLP.models.BiLSTMCRF 作为模型。

# In[3]:


from fastNLP.models import BiLSTMCRF
# 这是由于BiLSTMCRF模型的forward函数接受的words，而不是chars，所以需要把这一列重新命名
data_bundle.rename_field('chars', 'words')
model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200,
                  dropout=0.5,
                  target_vocab=data_bundle.get_vocab('target'))

# ## 进行训练
# 下面我们选择用来评估模型的metric，以及优化用到的优化函数。

# In[4]:


from fastNLP import SpanFPreRecMetric
from torch.optim import Adam
from fastNLP import LossInForward

metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
optimizer = Adam(model.parameters(), lr=1e-2)
loss = LossInForward()

# 使用Trainer进行训练, 您可以通过修改 device 的值来选择显卡。




from fastNLP import Trainer
import torch

device = 0 if torch.cuda.is_available() else 'cpu'
trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device)
trainer.train()

# ## 进行测试
# 训练结束之后过，可以通过 Tester 测试其在测试集上的性能




from fastNLP import Tester

tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
tester.test()

# ## 使用更强的Bert做序列标注
# 
# 在fastNLP使用Bert进行任务，您只需要把fastNLP.embeddings.StaticEmbedding 切换为 fastNLP.embeddings.BertEmbedding（可修改 device 选择显卡）。

# In[8]:


from fastNLP.io import WeiboNERPipe

data_bundle = WeiboNERPipe().process_from_file()
data_bundle.rename_field('chars', 'words')

from fastNLP.embeddings import BertEmbedding

embed = BertEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn')
model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200,
                  dropout=0.5,
                  target_vocab=data_bundle.get_vocab('target'))

from fastNLP import SpanFPreRecMetric
from torch.optim import Adam
from fastNLP import LossInForward

metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
optimizer = Adam(model.parameters(), lr=2e-5)
loss = LossInForward()

from fastNLP import Trainer
import torch

device = 5 if torch.cuda.is_available() else 'cpu'
trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=12,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device)
trainer.train()

from fastNLP import Tester

tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
tester.test()


