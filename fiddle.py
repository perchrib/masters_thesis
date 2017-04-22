from __future__ import print_function
import time
import os
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from nltk import sent_tokenize
import numpy as np


data = np.ones((2, 2), dtype=np.int64) * -1
print(data)
p = Parser()

text = ["@hola i have found something cool @hola http://c2.com/cgi/wiki?GeraldWeinbergQuotes",
        "Kids No #yo Longer http://feedly.com/k/103bzsz Good read", " Matt Taibbi - Everything Is Rigged : The Biggest Price-Fixing Scandal Ever no good readhttp :/ / www.rollingstone.com/politics/news/everything-is-rigged-the-biggest-financial-scandal-yet-20130425?print=true"]


# text = p.replace('url', 'U ', text)
# print(text)
#
# text = p.replace('@', 'M', text)
# print(text)
#
# text = p.replace('#', 'H', text)
# print(text)
#
text = p.replace_all(text)
print(text)


# Dataset statistics
# texts, labels, metadata, labels_index = prepare_dataset()
# x_train, y_train, meta_train, x_val, y_val, meta_val, word_index = format_dataset_word_level(texts, labels, metadata)
#
# display_gender_distribution(metadata)
# display_gender_distribution(meta_train)
# display_gender_distribution(meta_val)
#
# def foo():
#         a = 1
#         b = [2, 3]
#         return a, []
#
#
# def bar(model, extra_info, data):
#         print(model)
#         print(extra_info)
#         print(data)
# bar(*foo(), data="Data")
