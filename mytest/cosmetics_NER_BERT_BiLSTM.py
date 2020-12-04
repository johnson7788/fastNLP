#!/usr/bin/env python
# coding: utf-8
# # 序列标注
# ## 载入数据, 加载微博数据
from fastNLP.io import WeiboNERPipe

# ## 使用更强的Bert做序列标注
#
# 在fastNLP使用Bert进行任务，您只需要把fastNLP.embeddings.StaticEmbedding 切换为 fastNLP.embeddings.BertEmbedding（可修改 device 选择显卡）。

data_bundle = WeiboNERPipe().process_from_file(paths="data")
print(data_bundle.get_dataset('train')[:2])

data_bundle.rename_field('chars', 'words')

from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BiLSTMCRF

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

device = 0 if torch.cuda.is_available() else 'cpu'
trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=12,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device)
trainer.train()

from fastNLP import Tester

tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
tester.test()
