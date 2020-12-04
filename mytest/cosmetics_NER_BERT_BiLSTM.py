#!/usr/bin/env python
# coding: utf-8
# # 序列标注


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
    model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200,
                      dropout=0.5,
                      target_vocab=data_bundle.get_vocab('target'))

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
    # do_train(data_bundle, model, metric)
    do_test(data_bundle, metric, model_path="output")