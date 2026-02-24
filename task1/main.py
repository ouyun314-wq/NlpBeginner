import numpy
import csv
import random

from comparison_plot import comprehensive_comparison_plot
from constant import seed, test_rate, max_item
from feature import Bag
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg'


with open('..\\data\\sentiment-analysis-on-movie-reviews\\train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

data = temp[1:]

print("=" * 60)
print("综合对比实验：N-gram × 批次策略 × 学习率")
print("=" * 60)

# 构建不同ngram的bags
bags_dict = {}

# 1-gram
random.seed(seed)
numpy.random.seed(seed)
bags_dict["1-gram"] = Bag(data, test_rate, max_item, gram=1)

# (1,2)-gram
random.seed(seed)
numpy.random.seed(seed)
bags_dict["(1,2)-gram"] = Bag(data, test_rate, max_item, gram=2)

# (1,2,3)-gram
random.seed(seed)
numpy.random.seed(seed)
bags_dict["(1,2,3)-gram"] = Bag(data, test_rate, max_item, gram=3)

# 运行综合对比实验
random.seed(seed)
numpy.random.seed(seed)
comprehensive_comparison_plot(bags_dict, save_dir='result')
