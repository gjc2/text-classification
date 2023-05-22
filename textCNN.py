import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data.utils import get_tokenizer
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Truncate, PadTransform
import torch.utils.data as Data
import matplotlib.pyplot as plt

# define Network
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size[0], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size[1], embedding_dim))
        self.conv_3 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size[2], embedding_dim))
        self.fc = nn.Linear(len(filter_size)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        # print(embedded.size())
        conved_1 = nn.functional.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = nn.functional.relu(self.conv_2(embedded).squeeze(3))
        conved_3 = nn.functional.relu(self.conv_3(embedded).squeeze(3))

        pooled_1 = nn.functional.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = nn.functional.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = nn.functional.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1))
        return self.fc(cat)


# define parameter

EMBEDDING_DIM = 100
N_FILTERS = 10
FILTER_SIZE = [3, 4, 5]
OUTPUT_DIM = 4
DROPOUT = 0.3

# load data
# train_data = 'ag_news.train'
# test_data = 'ag_news.test'

# process data
tokenize = get_tokenizer("basic_english")


def yield_tokens(data):
    for _, text in data:
        yield tokenize(text)


train_set = datasets.AG_NEWS(root='.data', split='train')
test_set = datasets.AG_NEWS(root='.data', split='test')

vocab = build_vocab_from_iterator(yield_tokens(train_set), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab['<unk>'])

INPUT_DIM = len(vocab)


device = torch.device('cuda')


def createdataset(train_set):
    label_list, text_list = [], []
    truncate = Truncate(max_seq_len=200)
    pad = PadTransform(max_length=200, pad_value=vocab['<unk>'])
    for (_label, _text) in train_set:
        label_list.append([i+1 == _label for i in range(4)])
        text = vocab(tokenize(_text))
        text = truncate(text)
        text = torch.tensor(text, dtype=torch.int64)
        text = pad(text)
        text_list.append(text)

    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.vstack(text_list)

    return text_list.to(device), label_list.to(device),


batch_size = 128
input_data, input_label = createdataset(train_set)
ag_news_dataset = Data.TensorDataset(input_data, input_label)
test_d, test_l = createdataset(test_set)
ag_news_test = Data.TensorDataset(test_d, test_l)
loader = Data.DataLoader(ag_news_dataset, batch_size, True, drop_last=True)
test_loader = Data.DataLoader(ag_news_test, batch_size, True, drop_last=True)

print(type(loader))


model = Net(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZE, OUTPUT_DIM, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_data, batch_label in tqdm(loader):
        pred = model(batch_data).squeeze(1)
        # print(pred.shape, batch_label.shape)
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
            pred = model(batch_data).squeeze(1)
            loss = criterion(pred, batch_label)
            acc = accuracy(pred, batch_label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def accuracy(pred, y):

    _, pred = torch.max(pred, 1)
    _, y = torch.max(y, 1)
    correct = pred.detach().cpu().numpy() == y.detach().cpu().numpy()
    acc = np.sum(correct) / batch_size
    return acc


N_EPOCHS = 10


def train_model():
    best_valid_loss = float('inf')
    train_l, train_a, test_l, test_a = [], [], [], []
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        train_l.append(train_loss)
        train_a.append(train_acc)
        test_l.append(valid_loss)
        test_a.append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), f'checkpoints/textcnn_model_{epoch}.pt')
        print(f'Epoch : {epoch+1}, train_loss :{train_loss} ,train_acc = {train_acc}, test_loss:{valid_loss}, test_acc={valid_acc}')
    plt.figure
    plt.title('cnn network')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.plot([x for x in range(len(train_l))], train_l, 'r-', label="train_loss")
    plt.plot([x for x in range(len(train_a))], train_a, 'b-.', label="train_acc")
    plt.plot([x for x in range(len(test_l))], test_l, 'g-', label="valid_loss")
    plt.plot([x for x in range(len(test_a))], test_a, 'y-.', label="valid_acc")
    plt.legend()
    plt.savefig('cnn.png', dpi=1000, bbox_inches='tight')
    plt.show()

# model.load_state_dict(torch.load('textcnn_model.pt'))

# test_loss, test_acc = evaluate(model, test_loader, criterion)
# print(f'test loss:{test_loss},test acc:{test_acc}')

if __name__ == '__main__':
    train_model()




