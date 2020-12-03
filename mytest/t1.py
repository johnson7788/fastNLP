from fastNLP import Vocabulary
from fastNLP import DataSet

tr_data = DataSet({'chars': [
    ['今', '天', '心', '情', '很', '好', '。'],
    ['被', '这', '部', '电', '影', '浪', '费', '了', '两', '个', '小', '时', '。']
],
    'target': ['positive', 'negative']
})
dev_data = DataSet({'chars': [
    ['住', '宿', '条', '件', '还', '不', '错'],
    ['糟', '糕', '的', '天', '气', '，', '无', '法', '出', '行', '。']
],
    'target': ['positive', 'negative']
})

vocab = Vocabulary()
#  将验证集或者测试集在建立词表是放入no_create_entry_dataset这个参数中。
vocab.from_dataset(tr_data, field_name='chars', no_create_entry_dataset=[dev_data])
