import nltk
from nltk import AlignedSent
from nltk.translate import IBMModel1
from nltk.translate import Alignment
import os
import pickle

# f = open("data/multi30k/train.en")
# data = f.read()
# source_sentence = data.split("\n")
#
# f = open("data/multi30k/train.fr", "r", encoding= "utf-8")
# data = f.read()
# target_sentence = data.split("\n")
#
# bitext = []
# # Tạo các câu nguồn và đích để dịch
# for i in range(len(target_sentence)):
#     source_tokens = nltk.word_tokenize(source_sentence[i].lower())
#     target_tokens = nltk.word_tokenize(target_sentence[i].lower())
#     bitext.append(AlignedSent(source_tokens, target_tokens))
#
# # # Khởi tạo mô hình dịch IBM Model 1
# ibm1 = IBMModel1(bitext, 5)
#
# # Dịch câu nguồn sang câu đích
# translation_probabilities = ibm1.translation_table
# translation_dict = {}
#
# for s in ibm1.translation_table:
#     for t in ibm1.src_vocab:
#         probability = translation_probabilities[s][t]
#         if s not in translation_dict:
#             translation_dict[s] = {}
#         translation_dict[s][t] = probability
#
# with open(os.path.join("./model", "model.pkl"), "wb") as f:
#     pickle.dump(translation_dict,f)

with open("./model/model.pkl", "rb") as f:
    data = f.read()
f.close()
translation_dict = pickle.loads(data)
target_translation = []

test_sentence = "i am fine"
test = test_sentence.split(" ")
for source_word in test:
    targat_word = max(translation_dict[source_word], key=translation_dict[source_word].get)
    target_translation.append(targat_word)

target_sentence_translation = ' '.join(target_translation)

print(target_sentence_translation)
