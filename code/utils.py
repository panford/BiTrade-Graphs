from models import GCNet, GATNet, AGNNet, GAEncoder, VGAEncoder
import torch
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import jaccard_similarity_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GAE, VGAE, GATConv, AGNNConv, ChebConv
import csv

def makepath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def write_log(log, file_name):
    with open(file_name + ".csv", mode='w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        log_writer.writerow(log)
        log_writer.writerow(log)
    return

def eval_metrics(ytrue, ypred):
    jac = jaccard_similarity_score(ytrue, ypred)
    f_one = f1_score(ytrue, ypred)
    return jac, f_one

def getdata(infilepath):
    npzfile = np.load(infilepath, allow_pickle=True)
    node_attributes = npzfile['attr_data']
    attribute_shape = npzfile['attr_shape']
    trade_adj = npzfile['sparse_adj_trade']
    dist_adj = npzfile['sparse_adj_dists']
    class_labels = npzfile['labels']
    class_names = npzfile['class_names']

    trade_data_adj = trade_adj.tolist()
    trade_edge_attr = torch.tensor(trade_data_adj.data, dtype=torch.float32)
    tsrc, ttar = trade_data_adj.nonzero()[0], trade_data_adj.nonzero()[1]
    node_attributes = torch.tensor(node_attributes, dtype=torch.float32)
    trade_edge_index = torch.tensor([tsrc, ttar], dtype=torch.long)
    y = torch.tensor(class_labels, dtype=torch.long)

    n = len(node_attributes)
    test_size = int(n * 0.3)
    train_idx, test_idx = train_test_split(range(len(node_attributes)), test_size=test_size, random_state=42)
    trade_data = Data(x=node_attributes, y=y, edge_index=trade_edge_index, edge_attr=trade_edge_attr)
    test_size = int(len(trade_data.x) * 0.20)  # Use 70% for training and 30% for testing
    trade_data.train_idx = torch.tensor(train_idx, dtype=torch.long)
    trade_data.test_idx = torch.tensor(test_idx, dtype=torch.long)

    trade_data.train_mask = torch.cat((torch.zeros(test_size, dtype=torch.uint8),
                                       torch.ones(n - test_size, dtype=torch.uint8)))

    # trade_data.val_mask = torch.cat((torch.zeros(train_mask_size, dtype=torch.uint8),
    #                                  torch.ones(val_mask_size,dtype=torch.uint8),
    #                                  torch.zeros(test_mask_size ,dtype=torch.uint8)))

    trade_data.test_mask = torch.cat((torch.zeros(n - test_size, dtype=torch.uint8),
                                      torch.ones(test_size, dtype=torch.uint8)))

    trade_data.num_classes = trade_data.y.max() + 1

    return trade_data

def plot_classify(train_losses, test_losses, accs, output_dir, epochs, figname):

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(range(1, epochs + 1), train_losses, label='Train loss')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning curve during training and testing')

    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, epochs + 1), accs, label='Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('A plot of accuracy per epoch')
    plt.savefig(figname+".png")

def plot_linkpred(train_losses, test_losses, aucs, aps, output_dir, epochs, figname):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(range(1, epochs + 1), train_losses, label='Train loss')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction loss on train and test')
    ax1.set_title('Learning curve for the Graph autoencoder')

    ax2 = fig.add_subplot(122)
    ax2.plot(range(1, epochs + 1), aucs, label='AUC')
    ax2.plot(range(1, epochs + 1), aps, label='AP')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC / AP')
    ax2.set_title('AUCs and APs on test sets')
    plt.savefig(figname + ".png")

    return

def classifier_train_test(model_name, input_data, output_dir, epochs=1000, lr=0.01, weight_decay=0.0005):
    print('This is the type of the lr: ', type(lr))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: '.ljust(32), device)
    print('Model Name: '.ljust(32), str(model_name.__name__))
    print('Model params:{:19} lr: {}     weight_decay: {}'.format('', lr, weight_decay))
    print('Total number of epochs to run: '.ljust(32), epochs)
    print('*' * 65)
    data = input_data.clone().to(device)
    infeat = data.num_node_features
    outfeat = data.num_classes.item()

    model = model_name(infeat, outfeat).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    train_losses, test_losses = [], []
    accs = []
    best_val_acc = 0
    best_train_loss = 0
    best_val_loss = 0
    best_epoch = 0
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        test_loss = 0
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out[data.train_idx], data.y[data.train_idx])
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        test_out = model(data)
        # _ ,pred = model(data).max(dim = 1)
        test_loss = F.nll_loss(test_out[data.test_idx], data.y[data.test_idx])
        test_losses.append(test_loss)
        _, pred = test_out.max(dim=1)
        correct = float(pred[data.test_idx].eq(data.y[data.test_idx]).sum().item())
        acc = correct / len(data.test_idx)
        
        if best_val_acc < acc:
            best_val_acc = acc
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = test_loss

        accs.append(acc)

        figname = os.path.join(output_dir, "_".join((model_name.__name__, str(lr), str(weight_decay), str(epochs))))
        makepath(output_dir)

        if (epoch % int(epochs / 10) == 0):
            print('Epoch: {}           Train loss: {}   Test loss: {}    Test Accuracy: {}'.format(epoch, train_loss, test_loss, acc))
        if (epoch == epochs):
            print('-' * 65,
                  '\nFinal epoch: {}     Train loss: {}   Test loss: {}     Test Accuracy: {}'.format(epoch, train_loss, test_loss, acc))

    log = 'Best Epoch: {}, Train: {}, Val: {}, Test: {}'.format(best_epoch, best_train_loss, best_val_loss, best_val_acc)
    write_log(log, figname)


    print('-' * 65)
    print('\033[1mBest Accuracy\nEpoch: {}     Train loss: {}   Test loss: {}     Test Accuracy: {}\n'
          .format(best_epoch, best_train_loss, best_val_loss, best_val_acc))


    write_log(log, figname)
    plot_classify(train_losses, test_losses, accs, output_dir, epochs, figname)
    
    return acc


def run_GAE(input_data, output_dir, epochs=1000, lr=0.01, weight_decay=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: '.ljust(32), device)
    print('Model Name: '.ljust(32), 'GAE')
    print('Model params:{:19} lr: {}   weight_decay: {}'.format('', lr, weight_decay))
    print('Total number of epochs to run: '.ljust(32), epochs)
    print('*' * 70)

    data = input_data.clone().to(device)
    in_channels = data.num_features
    out_channels = data.num_classes.item()
    model = GAE(GAEncoder(in_channels, out_channels)).to(device)
    data = input_data.clone().to(device)
    split_data = model.split_edges(data)
    x, train_pos_edge_index, edge_attr = split_data.x.to(device), split_data.train_pos_edge_index.to(
        device), data.edge_attr.to(device)
    split_data.train_idx = split_data.test_idx = data.y = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses, test_losses = [], []
    aucs = []
    aps = []
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        test_loss = 0
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        train_loss = model.recon_loss(z, train_pos_edge_index)
        train_losses.append(train_loss)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        auc, ap = model.test(z, split_data.test_pos_edge_index, split_data.test_neg_edge_index)
        test_loss = model.recon_loss(z, data.test_pos_edge_index)
        test_losses.append(test_loss.item())
        aucs.append(auc)
        aps.append(ap)

        figname = os.path.join(output_dir, "_".join((GAE.__name__, str(lr), str(weight_decay))))
        makepath(output_dir)

        if (epoch % int(epochs / 10) == 0):
            print('Epoch: {}       Train loss: {}    Test loss: {}     AUC: {}    AP: {}'.format(epoch,train_loss,test_loss, auc, ap))
        if (epoch == epochs):
            print('-' * 65,
                  '\nFinal epoch: {}    Train loss: {}    Test loss: {}    AUC: {}    AP: {}'.format(
                      epoch, train_loss, test_loss, auc, ap))
        log = 'Final epoch: {}    Train loss: {}    Test loss: {}    AUC: {}    AP: {}'.format(
            epoch, train_loss, test_loss, auc, ap)
        write_log(log, figname)
    print('-' * 65)


    plot_linkpred(train_losses, test_losses, aucs, aps, output_dir, epochs, figname)
    return


def run_VGAE(input_data, output_dir, epochs=1000, lr=0.01, weight_decay=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: '.ljust(32), device)
    print('Model Name: '.ljust(32), 'VGAE')
    print('Model params:{:19} lr: {}     weight_decay: {}'.format('', lr, weight_decay))
    print('Total number of epochs to run: '.ljust(32), epochs)
    print('*' * 70)

    data = input_data.clone().to(device)
    model = VGAE(VGAEncoder(data.num_features, data.num_classes.item())).to(device)
    data = model.split_edges(data)
    x, train_pos_edge_index, edge_attr = data.x.to(device), data.train_pos_edge_index.to(device), data.edge_attr.to(
        device)
    data.train_idx = data.test_idx = data.y = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_losses = []
    test_losses = []
    aucs = []
    aps = []
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = 0, 0
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        train_loss = model.recon_loss(z, train_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        test_loss = model.recon_loss(z, data.test_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()
        test_losses.append(test_loss.item())
        aucs.append(auc)
        aps.append(ap)
        makepath(output_dir)
        figname = os.path.join(output_dir, "_".join((VGAE.__name__, str(lr), str(weight_decay), str(epochs))))
        # print('AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
        if (epoch % int(epochs / 10) == 0):
            print('Epoch: {}        Train loss: {}    Test loss: {}    AUC: {}    AP: {:.4f}'.format(epoch,
                                                                                                                 train_loss,
                                                                                                                 test_loss,
                                                                                                                 auc,
                                                                                                                 ap))
        if (epoch == epochs):
            print('-' * 65,
                  '\nFinal epoch: {}  Train loss: {}    Test loss: {}    AUC: {}    AP: {}'.format(
                      epoch, train_loss, test_loss, auc, ap))
        log = 'Final epoch: {}    Train loss: {}    Test loss: {}    AUC: {}    AP: {}'.format(
                epoch, train_loss, test_loss, auc, ap)
        write_log(log, figname)
    print('-' * 65)

    plot_linkpred(train_losses, test_losses, aucs, aps, output_dir, epochs, figname)
    return


    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax1.plot(range(1, epochs + 1), train_losses, label='Train loss')
    # ax1.plot(range(1, epochs + 1), test_losses, label='Test loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Learning curve during training and testing')
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(range(1, epochs + 1), accs, label='Accuracy')
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('Accuracy')
    # ax2.set_title('A plot of accuracy per epoch')
    # os.mkdir(output_dir, exist_ok = True)
    # figname = os.path.join(output_dir, "_".join((model_name.__name__,str(lr), str(weight_decay))))
    # plt.savefig(figname+".png")
    # plt.show()


    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax1.plot(range(1, epochs + 1), train_losses, label='Train loss')
    # ax1.plot(range(1, epochs + 1), test_losses, label='Test loss')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Reconstruction loss on train and test')
    # ax1.set_title('Learning curve for the Graph autoencoder')
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(range(1, epochs + 1), aucs, label='AUC')
    # ax2.plot(range(1, epochs + 1), aps, label='AP')
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('AUC / AP')
    # ax2.set_title('AUCs and APs on test sets')
    # figname = os.path.join(output_dir, "_".join((model_name.__name__, str(lr), str(weight_decay))))
    # plt.savefig(figname+".png")


    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax1.plot(range(1, epochs + 1), train_losses, label='Train loss')
    # ax1.plot(range(1, epochs + 1), test_losses, label='Test loss')
    # ax1.set_xlabel('Epochs')
    #
    # ax1.set_ylabel('Reconstruction loss')
    # ax1.set_title('Learning curve for the Variational Graph autoencoder')
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(range(1, epochs + 1), aucs, label='AUC')
    # ax2.plot(range(1, epochs + 1), aps, label='Average Precision score')
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('AUC / AP')
    # ax2.set_title('AUCs and Average Precision scores on test sets')
    # os.mkdir(output_dir)
    # figname = os.path.join(output_dir, "_".join((model_name, str(lr), str(weight_decay))))
    # plt.savefig(figname+".png")
    #
    