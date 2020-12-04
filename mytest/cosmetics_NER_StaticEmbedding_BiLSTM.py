#!/usr/bin/env python
# coding: utf-8
# # 序列标注
# ## 载入数据, 加载微博数据
from fastNLP.io import WeiboNERPipe

data_bundle = WeiboNERPipe().process_from_file(paths="data")
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
