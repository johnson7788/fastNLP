#!/usr/bin/env python
# coding: utf-8
# # 序列标注，纯BERT

import torch
import torch.nn as nn

from fastNLP.models.base_model import BaseModel
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules import decoder, encoder
from fastNLP.core.const import Const as C

class MySeqLabeling(BaseModel):
    r"""
    一个基础的Sequence labeling的模型。
    用于做sequence labeling的基础类。结构包含一层Embedding，一层FC，以及一层CRF。

    """

    def __init__(self, embed, num_classes):
        r"""

        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, embedding, ndarray等则直接使用该值初始化Embedding
        :param int hidden_size: LSTM隐藏层的大小
        :param int num_classes: 一共有多少类
        """
        super(MySeqLabeling, self).__init__()

        self.embedding = get_embeddings(embed)
        self.fc = nn.Linear(self.embedding.embedding_dim, num_classes)
        self.crf = decoder.ConditionalRandomField(num_classes)

    def forward(self, words, seq_len, target):
        r"""
        :param torch.LongTensor words: [batch_size, max_len]，序列的index
        :param torch.LongTensor seq_len: [batch_size,], 这个序列的长度
        :param torch.LongTensor target: [batch_size, max_len], 序列的目标值
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                    If truth is not None, return loss, a scalar. Used in training.
        """
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))
        x = self.embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x = self.fc(x)
        # [batch_size, max_len, num_classes]
        return {C.LOSS: self._internal_loss(x, target, mask)}

    def predict(self, words, seq_len):
        r"""
        用于在预测时使用

        :param torch.LongTensor words: [batch_size, max_len]
        :param torch.LongTensor seq_len: [batch_size,]
        :return: {'pred': xx}, [batch_size, max_len]
        """
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))

        x = self.embedding(words)
        # [batch_size, max_len, hidden_size * direction]
        x = self.fc(x)
        # [batch_size, max_len, num_classes]
        pred = self._decode(x, mask)
        return {C.OUTPUT: pred}

    def _internal_loss(self, x, y, mask):
        r"""
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        total_loss = self.crf(x, y, mask)
        return torch.mean(total_loss)

    def _decode(self, x, mask):
        r"""
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :return prediction: [batch_size, max_len]
        """
        tag_seq, _ = self.crf.viterbi_decode(x, mask)
        return tag_seq


def load_data():
    # ## 载入数据, 加载微博数据
    from fastNLP.io import WeiboNERPipe

    # ## 使用更强的Bert做序列标注
    #
    # 在fastNLP使用Bert进行任务，您只需要把fastNLP.embeddings.StaticEmbedding 切换为 fastNLP.embeddings.BertEmbedding（可修改 device 选择显卡）。

    data_bundle = WeiboNERPipe().process_from_file(paths="data")
    print(data_bundle.get_dataset('train')[:2])

    data_bundle.rename_field('chars', 'words')
    return data_bundle


def build_model_metric(data_bundle):

    from fastNLP.embeddings import BertEmbedding
    from fastNLP.models import BiLSTMCRF
    from fastNLP import SpanFPreRecMetric

    embed = BertEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn')
    model = MySeqLabeling(embed=embed, num_classes=len(data_bundle.get_vocab('target')))

    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))

    return model,metric

def do_train(data_bundle, model, metric):
    """
    BERT 模型效果比BiLSTM会好一些
    :param data_bundle:
    :param model:
    :param metric:
    :return:
    """
    from torch.optim import Adam
    from fastNLP import LossInForward
    from fastNLP import Trainer
    import torch

    optimizer = Adam(model.parameters(), lr=2e-5)
    loss = LossInForward()

    device = 0 if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=12,n_epochs=10,
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device, save_path="output")
    trainer.train()

def do_test(data_bundle, metric, model_path, save_excel="test.xlsx"):
    # ## 进行测试
    # 训练结束之后过，可以通过 Tester 测试其在测试集上的性能
    from fastNLP import Tester
    from fastNLP.io import ModelLoader
    import os
    #如果是一个目录，只用其中的一个模型
    if os.path.isdir(model_path):
        models_file = os.listdir(model_path)
        if len(models_file) != 1:
            print("模型文件不仅一个，请手动给定")
            import sys
            sys.exit(1)
        else:
            model_path = os.path.join(model_path,models_file[0])
    model = ModelLoader.load_pytorch_model(model_path)
    tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
    eval_results = tester.test()
    id2labels = data_bundle.vocabs['target'].idx2word
    test_contents = data_bundle.get_dataset('test').get_field("raw_chars").content
    true_labels = data_bundle.get_dataset('test').get_field("target").content
    predict_ids = eval_results['predict_results']
    results = []
    for content, true_id, predict_id in zip(test_contents, true_labels, predict_ids):
        label = list(map(lambda x: id2labels[x], true_id))
        predict = list(map(lambda x: id2labels[x], predict_id))
        if len(content) != len(label):
            print("句子内容和真实label长度不匹配，错误")
            print(content)
            print(label)
            break
        predict = predict[:len(label)]
        con = " ".join(content)
        la = " ".join(label)
        pre = " ".join(predict)
        print("句子:", con)
        print("真实标签:", la)
        print("预测标签:", pre)
        words = []
        word = ""
        for idx, p in enumerate(predict):
            if p.startswith('B-'):
                if word != "":
                    #说明上一个单词已经是一个完整的词了, 加到词表，然后重置
                    words.append(word)
                    word = ""
                word += content[idx]
            elif p.startswith('I-'):
                word += content[idx]
            else:
                #如果单词存在，那么加到词语表里面
                if word:
                    words.append(word)
                    word = ""
        print("真实的词:", words)
        results.append({'content':con, "words":words, "predict":pre})
    if save_excel:
        import pandas as pd
        df = pd.DataFrame(results)
        writer = pd.ExcelWriter(save_excel)
        df.to_excel(writer)
        writer.save()

if __name__ == '__main__':
    data_bundle = load_data()
    model, metric = build_model_metric(data_bundle)
    do_train(data_bundle, model, metric)
    # do_test(data_bundle, metric, model_path="output")