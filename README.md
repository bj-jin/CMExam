# CMExam
./zhwiki    存放中文wiki语料库（及预处理文件）

./build_index.sh    基于./zhwiki中预处理后的文件构建索引，并将索引文件存放在./index路径下

./index    用于存放./build_index.sh创建的索引

./data/data_proc.ipynb    基于./index中的索引，将数据作为query，为训练/验证/测试集检索wiki知识，从而构造训练/验证/测试集



首先需要下载中文语料库至./zhwiki路径下，然后运行./build_index.sh创建索引，然后使用./data/data_proc.ipynb构造带有wiki知识的数据集存放至./data下
