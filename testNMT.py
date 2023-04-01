import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy

# Sử dụng thư viện spaCy để tách từ
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# Định nghĩa các trình tạo dữ liệu
SRC = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)

# Tải dữ liệu từ bộ dữ liệu Multi30k
train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.de'), fields=(SRC, TRG), root='data')

# Xây dựng từ điển và chỉ số hóa dữ liệu
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Định nghĩa các thông số huấn luyện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)

# Định nghĩa một số hằng số
MAX_LENGTH = 50
SOS_token = TRG.vocab.stoi['<sos>']
EOS_token = TRG.vocab.stoi['<eos>']

# Định nghĩa lớp mã hóa
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Định nghĩa lớp giải mã
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
class Seq2Seq(nn.Module):
    def init(self, encoder, decoder, device):
        super(Seq2Seq, self).init()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder.initHidden()
        for i in range(max_len):
            output, hidden = self.encoder(src[i], hidden)
            output, hidden = self.decoder(trg[i], hidden)
            outputs[i] = output
        return outputs

    def translate_sentence(self, sentence, src_field, trg_field, max_len=50):
        self.eval()
        with torch.no_grad():
            tokens = [token.lower() for token in sentence]
            src_indexes = [src_field.vocab.stoi[token] for token in tokens]
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(self.device)
            src_len = torch.LongTensor([len(src_indexes)])
            hidden = self.encoder.initHidden()
            for i in range(len(tokens)):
                output, hidden = self.encoder(src_tensor[i], hidden)
            trg_indexes = [trg_field.vocab.stoi['<sos>']]
            for i in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                output, hidden = self.decoder(trg_tensor, hidden)
                pred_token = output.argmax(dim=1).item()
                trg_indexes.append(pred_token)
                if pred_token == trg_field.vocab.stoi['<eos>']:
                    break
            trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
            return trg_tokens[1:]

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_SIZE = 256
LEARNING_RATE = 0.01
encoder = EncoderRNN(INPUT_DIM, HIDDEN_SIZE).to(device)
decoder = DecoderRNN(HIDDEN_SIZE, OUTPUT_DIM).to(device)
model = Seq2Seq(encoder, decoder, device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

N_EPOCHS = 10
CLIP = 1
for epoch in range(N_EPOCHS):
    for i, batch in enumerate(train_iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])

sentence = "I love you"
tokens = [token.lower() for token in sentence.split()]
src_indexes = [SRC.vocab.stoi[token] for token in tokens]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

# Dịch câu tiếng Anh sang tiếng Pháp
translation = model.translate_sentence(src_tensor, SRC, TRG)
translation = " ".join(translation)
print(f"Câu tiếng Pháp tương ứng với câu tiếng Anh '{sentence}' là: {translation}")