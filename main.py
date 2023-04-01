import nltk
from nltk import AlignedSent
from nltk.translate import IBMModel1
from nltk.translate import Alignment

# Tạo các câu nguồn và đích để dịch
source_sentence = ["I am happy", "i am beautiful", "I go to school"]
target_sentence = ["Je suis heureux", "Je suis beau", "je vais à l'école"]

# Tiền xử lý các câu nguồn và đích bằng cách tách từ
source_tokens = nltk.word_tokenize(source_sentence[0].lower())
target_tokens = nltk.word_tokenize(target_sentence[0].lower())

bitext = []
bitext.append(AlignedSent(source_tokens, target_tokens))

source_tokens = nltk.word_tokenize(source_sentence[1].lower())
target_tokens = nltk.word_tokenize(target_sentence[1].lower())

bitext.append(AlignedSent(source_tokens, target_tokens))

source_tokens = nltk.word_tokenize(source_sentence[2].lower())
target_tokens = nltk.word_tokenize(target_sentence[2].lower())

bitext.append(AlignedSent(source_tokens, target_tokens))
# # Khởi tạo mô hình dịch IBM Model 1
ibm1 = IBMModel1(bitext, 5)

# Dịch câu nguồn sang câu đích
translation_probabilities = ibm1.translation_table
translation_dict = {}

for s in ibm1.translation_table:
    for t in ibm1.src_vocab:
        #print(s, t, translation_probabilities[s][t])
        probability = translation_probabilities[s][t]
        if s not in translation_dict:
            translation_dict[s] = {}
        translation_dict[s][t] = probability



target_translation = []
test = ["i","am", "happy"]
for source_word in test:
    targat_word = max(translation_dict[source_word], key=translation_dict[source_word].get)
    target_translation.append(targat_word)

target_sentence_translation = ' '.join(target_translation)

print(target_sentence_translation)
