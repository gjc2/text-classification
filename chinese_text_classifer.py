import json
import jieba
import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from torchtext.data.utils import get_tokenizer
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Truncate, PadTransform
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1 define network(transformer)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, n_head, n_layers):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            vocab_size=vocab_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            batch_first=True,
            nhead = n_head,

        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
         text = self.embedding(text)
         text = self.pos_encoder(text)
         text = self.transformer(text)
         text = text.mean(dim=1)
         text = self.fc(text)
         return text


# 2 define parameter

VOCAB_SIZE = 0
EMBEDDING_DIM = 100
OUTPUT_DIM = 10
DROPOUT = 0.3
BATCH_SIZE = 16
N_HEAD = 2
N_LAYERS = 3


# 3 process_data


train_set = []
valid_set = []
categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
categories = [x for x in categories]
cat_to_id = dict(zip(categories, range(len(categories))))
vocab = {}
device = torch.device('cuda')
print('load data')
with open('cnews/cnews.train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        label, content = line.strip().split('\t')
        train_set.append((label, content))
with open('cnews/cnews.val.txt', 'r', encoding='utf-8') as f:
    for line in f:
        label, content = line.strip().split('\t')
        valid_set.append((label, content))
print('load vocab')
with open('cnews/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

# def show_data(data_set):
#     print(1)
#     text_len = [len([word for word in jieba.cut(x)]) for _, x in tqdm(data_set)]
#     print(max(text_len))
#     plt.hist(text_len, bins=1000, density=True, cumulative=True, color='red')
#     plt.show()
#
#
# show_data(train_set)


def create_dataset(_set):
    max_length = 1000
    label_list, text_list = [], []
    for _label, _text in _set:
        label_list.append([i == cat_to_id[_label] for i in range(10)])
        text = [word for word in jieba.cut(_text)]
        text2id = []
        if len(text) >= max_length:
            text = text[0:max_length]
        for word in text:
            if vocab.get(word, 0) == 0:
                text2id.append(vocab['<unk>'])
            else:
                text2id.append(vocab[word])
        while len(text2id) < max_length:
            text2id.append(vocab['<pad>'])
        text2id = torch.tensor(text2id, dtype=torch.int64)
        text_list.append(text2id)
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.vstack(text_list)
    return text_list.to(device), label_list.to(device)


print('create train data')
input_data, input_label = create_dataset(train_set)
ag_news_train = Data.TensorDataset(input_data, input_label)
print('create test data')
test_data, test_label = create_dataset(valid_set)
ag_news_test = Data.TensorDataset(test_data, test_label)
print('create data loader')
train_loader = Data.DataLoader(ag_news_train, BATCH_SIZE, True, drop_last=True)
test_loader = Data.DataLoader(ag_news_test, BATCH_SIZE, True, drop_last=True)

# 4 create model

model = Net(VOCAB_SIZE, EMBEDDING_DIM, OUTPUT_DIM, N_HEAD, N_LAYERS).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())

# 5 train model


def accuracy(pred, label):
    _, pred = torch.max(pred, 1)
    _, label = torch.max(label, 1)
    correct = pred.detach().cpu().numpy() == label.detach().cpu().numpy()
    acc = np.sum(correct) / BATCH_SIZE
    return acc


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_data, batch_label in tqdm(loader):
        pred = model(batch_data)
        # print(pred.shape) # [batch_size, 4]
        loss = criterion(pred, batch_label)
        acc = accuracy(pred, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_acc += acc.item()
        epoch_loss += loss.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch_data, batch_label in tqdm(loader):
            pred = model(batch_data)
            loss = criterion(pred, batch_label)
            acc = accuracy(pred, batch_label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


N_EPOCHS = 10


def train_model():
    best_valid_loss = float('inf')
    train_l, train_a, test_l, test_a = [], [], [], []
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        train_l.append(train_loss)
        train_a.append(train_acc)
        test_l.append(valid_loss)
        test_a.append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), f'checkpoints/linear/linear_model_{epoch}.pt')
        print(
            f'Epoch : {epoch + 1}, train_loss :{train_loss} ,train_acc = {train_acc}, test_loss:{valid_loss}, test_acc={valid_acc}')
    plt.figure
    plt.title('transformer-chinese')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.plot([x for x in range(len(train_l))], train_l, 'r-', label="train_loss")
    plt.plot([x for x in range(len(train_a))], train_a, 'b-.', label="train_acc")
    plt.plot([x for x in range(len(test_l))], test_l, 'g-', label="valid_loss")
    plt.plot([x for x in range(len(test_a))], test_a, 'y-.', label="valid_acc")
    plt.legend()
    plt.savefig('transformer-chinese.png', dpi=1000, bbox_inches='tight')
    plt.show()
# model.load_state_dict(torch.load('/'))


if __name__ == '__main__':
    train_model()
