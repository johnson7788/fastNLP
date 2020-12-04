#!/usr/bin/env python
# coding: utf-8
# # 序列标注
# ## 载入数据, 加载微博数据

def load_data():
    from fastNLP.io import WeiboNERPipe

    data_bundle = WeiboNERPipe().process_from_file(paths="data")
    print(data_bundle.get_dataset('train')[:2])
    return data_bundle


def build_model_metric(data_bundle):
    # ## 模型构建
    #
    # 首先选择需要使用的Embedding类型。关于Embedding的相关说明可以参见《使用Embedding模块将文本转成向量》。 在这里我们使用通过word2vec预训练的中文汉字embedding。

    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import SpanFPreRecMetric

    embed = StaticEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name='cn-char-fastnlp-100d')

    # 选择好Embedding之后，我们可以使用fastNLP中自带的 fastNLP.models.BiLSTMCRF 作为模型。

    # In[3]:

    from fastNLP.models import BiLSTMCRF

    # 这是由于BiLSTMCRF模型的forward函数接受的words，而不是chars，所以需要把这一列重新命名
    data_bundle.rename_field('chars', 'words')
    model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200,
                      dropout=0.5,
                      target_vocab=data_bundle.get_vocab('target'))
    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
    return model, metric

def do_train(data_bundle, model, metric):


    # ## 进行训练
    # 下面我们选择用来评估模型的metric，以及优化用到的优化函数。


    from torch.optim import Adam
    from fastNLP import LossInForward


    optimizer = Adam(model.parameters(), lr=1e-2)
    loss = LossInForward()

    # 使用Trainer进行训练, 您可以通过修改 device 的值来选择显卡。

    from fastNLP import Trainer
    import torch

    device = 0 if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer,
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric, device=device, save_path="output", n_epochs=50)
    trainer.train()

def do_test(data_bundle,model, metric):
    # ## 进行测试
    # 训练结束之后过，可以通过 Tester 测试其在测试集上的性能

    from fastNLP import Tester

    tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric)
    eval_results = tester.test()
    id2labels = data_bundle.vocabs['target'].idx2word
    test_contents = data_bundle.get_dataset('test').get_field("raw_chars").content
    true_labels = data_bundle.get_dataset('test').get_field("target").content
    predict_ids = eval_results['predict_results']
    for content, true_id, predict_id in zip(test_contents, true_labels, predict_ids):
        label = list(map(lambda x: id2labels[x], true_id))
        predict = list(map(lambda x: id2labels[x], predict_id))
        if len(content) != len(label):
            print("句子内容和真实label长度不匹配，错误")
            print(content)
            print(label)
            break
        predict = predict[:len(label)]
        print("句子:", " ".join(content))
        print("真实标签:", " ".join(label))
        print("预测标签:", " ".join(predict))


if __name__ == '__main__':
    data_bundle = load_data()
    model, metric = build_model_metric(data_bundle)
    # do_train(data_bundle, model, metric)
    do_test(data_bundle, model, metric)