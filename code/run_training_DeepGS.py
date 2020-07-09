import pickle
import sys
import timeit

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error, precision_score, recall_score
from emetrics import get_aupr, get_cindex, get_rm2

from gensim.models import word2vec
from keras.preprocessing import text, sequence

import pandas as pd
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        # dim=10
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        # predefined word embedding
        self.embed_word = nn.Embedding(10000, 100)
        self.embed_word.weight = nn.Parameter(torch.tensor(pro_embedding_matrix, dtype=torch.float32))
        self.embed_word.weight.requires_grad = True

        self.embed_smile = nn.Embedding(100, 100)
        self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
        self.embed_smile.weight.requires_grad = True

        # define 3 dense layer
#         self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
#                                     for _ in range(layer_gnn)])
        #GCN
        self.gcn1 = GATConv(dim, dim, heads=10, dropout=0.2)
        self.gcn2 = GATConv(dim * 10, 128, dropout=0.2)
        self.fc_g1 = nn.Linear(128, 128)
        
        
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        
#         self.W_rnn = nn.GRU(bidirectional=False, num_layers=2, input_size=100, hidden_size=100)
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)

        self.W_attention = nn.Linear(dim, 100)
        self.P_attention = nn.Linear(100, 100)
#         self.W_out = nn.ModuleList([nn.Linear(2*100+128, 2*100+128)
#                                     for _ in range(layer_output)])
#         # self.W_interaction = nn.Linear(2*dim, 2)
#         self.W_interaction = nn.Linear(2*100+128, 1)
        self.W_out = nn.ModuleList([nn.Linear(100+200+128, 100+200+128)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(100+200+128, 1)
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer_cnn):
            xs = self.W_cnn[i](xs)
            xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
#         del Atteintion
#         h = torch.relu(self.W_attention(x))
#         hs = torch.relu(self.P_attention(xs))
#         weights = torch.tanh(F.linear(h, hs))
#         ys = torch.t(weights) * hs

#         return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    
    def rnn(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
        
    def forward(self, inputs):

        fingerprints, adjacency, words, smiles = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
#         compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        adjacency = adjacency.transpose(1, 0)
        x = F.dropout(fingerprint_vectors, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, adjacency))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, adjacency)
        x = self.relu(x)
        batch = torch.LongTensor([0]*len(x)).to(device)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        compound_vector = self.relu(x)
        

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(word_vectors, layer_cnn)
#         protein_vector = self.rnn(word_vectors, layer_cnn)

        """smile vector with attention-CNN."""
        # add the feature of word embedding of SMILES string
        smile_vectors = self.embed_smile(smiles)
        after_smile_vectors = self.rnn(smile_vectors)

        """Concatenate the above two vectors and output the interaction."""
        # concatenate with three types of features
        cat_vector = torch.cat((compound_vector, protein_vector, after_smile_vectors), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        predicted_interaction = torch.squeeze(predicted_interaction, 0)
        if train:
            loss_fuc = torch.nn.MSELoss()
            loss = loss_fuc(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            ys = predicted_interaction.to('cpu').data.numpy()
            # predicted_labels = list(map(lambda x: np.argmax(x), ys))
            # predicted_scores = list(map(lambda x: x[1], ys))
            predicted_scores = ys
            return correct_labels, predicted_scores
        
        
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, S = [], []
        for data in dataset:
            (correct_labels,predicted_scores) = self.model(data, train=False)
            # print(correct_labels, predicted_scores)
            T.append(correct_labels)
            S.append(predicted_scores)
        MSE = mean_squared_error(T, S)
        cindex = get_cindex(T, S)
        rm2 = get_rm2(T, S)
        AUPR = get_aupr(T, S)
        return MSE, cindex, rm2, AUPR, T, S

    def save_MSEs(self, MSEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MSEs)) + '\n')
            
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Predict\n')
            f.write(str(predictions))
            
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle = True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def w2v_pad(protein, maxlen_,victor_size):

    #keras API 
    tokenizer = text.Tokenizer(num_words=10000, lower=False,filters="　")
    tokenizer.fit_on_texts(protein)
    protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(protein), maxlen=maxlen_)

    word_index = tokenizer.word_index
    nb_words = len(word_index)
    print(nb_words)
    protVec_model = {}
    with open("../dataset/embed/protVec_100d_3grams.csv", encoding='utf8') as f:
        for line in f:
            values = eval(line).rsplit('\t')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            protVec_model[word] = coefs
    print("add protVec finished....")


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=protVec_model[word] if word in protVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del protVec_model
    print(embedding_matrix.shape)
    return protein_, word_index, embedding_matrix

def smile_w2v_pad(smile, maxlen_,victor_size):

    #keras API
    tokenizer = text.Tokenizer(num_words=100, lower=False,filters="　")
    tokenizer.fit_on_texts(smile)
    smile_ = sequence.pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_)

    word_index = tokenizer.word_index
    nb_words = len(word_index)
    print(nb_words)
    smileVec_model = {}
    with open("../dataset/embed/Atom.vec", encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    print("add smileVec finished....")


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    print(embedding_matrix.shape)
    return smile_, word_index, embedding_matrix


if __name__ == "__main__":
    torch.cuda.set_device(0)
    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    """processing protein sequence"""
    with open('../dataset/' + DATASET + '/proteins.txt', 'r') as f:
        protein = f.read().strip().split('\n')

    """processing SMILES"""
    with open('../dataset/' + '/smile_n_gram.txt', 'r') as f:
        smile = f.read().strip().split('\n')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/inputtest/'
                 'radius' + radius + '_ngram' + ngram + '/')

    protein_, pro_word_index, pro_embedding_matrix = w2v_pad(protein, 2000, 100)
    np.save(dir_input+"protein.npy", protein_)
    del protein_


    smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smile, 100, 100)
    np.save(dir_input+"smile.npy", smile_)
    del smile_


    smiles = load_tensor(dir_input + 'smile', torch.LongTensor)
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.LongTensor)
    proteins = load_tensor(dir_input + 'protein', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.FloatTensor)
    Y = load_tensor(dir_input + 'interactions', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """data preprocessing"""
    Y = np.asarray(Y)
    interactions = -(np.log10(Y / (math.pow(10, 9))))
    interactions = list(interactions)


    # proteins = torch.from_numpy(protein_).long().to(device)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, smiles, interactions))
    dataset = shuffle_dataset(dataset, 1234)

    dataset_train, dataset_ = split_dataset(dataset, 2/3)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    """Set a model."""
#     file_model = '../output/model/kiba--radius2--ngram3--dim32--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-4--lr_decay0.9--decay_interval20--weight_decay1e-5--iteration100'
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
#     model.load_state_dict(torch.load(file_model))
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MSEs = '../output/result/' + setting + 'GAT_bigru_cnn.txt'
    file_T = '../output/predict/T--' + setting + 'GAT_bigru_cnn.txt'
    file_S = '../output/predict/S--' + setting + 'GAT_bigru_cnn.txt'
    file_model = '../output/model/' + setting + 'GAT_bigru_cnn'
    MSEs = ('Epoch\tTime(sec)\tLoss_train\tMSE_dev\t'
            'MSE_test\tciindex_test\trm2\tAUPR')
    with open(file_MSEs, 'w') as f:
        f.write(MSEs + '\n')

    """Start training."""
    print('Training...')
    print(MSEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MSE_dev = tester.test(dataset_dev)[0]
        MSE_test, cindex, rm2, AUPR, T, S = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MSEs = [epoch, time, loss_train, MSE_dev,
                MSE_test,cindex, rm2, AUPR]
        tester.save_MSEs(MSEs, file_MSEs)
        tester.save_model(model, file_model)
        tester.save_predictions(T, file_T)
        tester.save_predictions(S, file_S)
        print('\t'.join(map(str, MSEs)))
