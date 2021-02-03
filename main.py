import argparse
import pickle
import time

import torch
import numpy as np
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, classification_report

from GraphDataset import GraphDataset
from model import Model

start_time = time.time()


def train(args, model, optimizer, dataset, gd):
    model.train()
    one_epoch = 0
    # labels = []
    # preds = []

    gd.files_shuffle()
    loss_accum = 0
    while one_epoch == 0:
        attri_descriptors, adj_mats, label, reconstructs, one_epoch = gd.data_gen(dataset, args.batch_size)
        attri_descriptors = list(zip(*attri_descriptors))
        class_capsule_output, loss, margin_loss, reconstruction_loss, label, pred = model(
            adj_mats, attri_descriptors, label, reconstructs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # labels.append(label.detach())
        # preds.append(pred.detach())
        loss_accum += loss.detach().cpu().item()
    # labels = torch.cat(labels)
    # preds = torch.cat(preds)
    # print('loss', loss_accum)
    return loss_accum


def test(args, model, dataset, gd, split):

    model.eval()
    one_epoch = 0
    labels = []
    preds = []
    while one_epoch == 0:
        attri_descriptors, adj_mats, label, reconstructs, one_epoch = gd.data_gen(dataset, args.batch_size)
        attri_descriptors = list(zip(*attri_descriptors))
        with torch.no_grad():
            class_capsule_output, loss, margin_loss, reconstruction_loss, label, pred = model(
                adj_mats, attri_descriptors, label, reconstructs)
        labels.append(label.detach().cpu())
        preds.append(pred.detach().cpu())
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    accuracy = accuracy_score(labels.numpy(), preds.numpy())
    # print(split, 'accuracy', accuracy)
    return accuracy


def main():
    parser = argparse.ArgumentParser("GraphClassification")
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument("--dataset_dir", type=str, default="data_plk/ENZYMES", help="Where dataset is stored.")
    parser.add_argument("--epochs", type=int, default=3000, help="epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--iterations", type=int, default=3, help="number of iterations of dynamic routing")
    parser.add_argument("--seed", type=int, default=12345, help="Initial random seed")
    parser.add_argument("--node_embedding_size", default=8, type=int,
                        help="Intended subgraph embedding size to be learnt")
    parser.add_argument("--graph_embedding_size", default=8, type=int,
                        help="Intended graph embedding size to be learnt")
    parser.add_argument("--num_gcn_channels", default=2, type=int,
                        help="Number of channels at each layer")
    parser.add_argument("--num_gcn_layers", default=5, type=int,
                        help="Number of GCN layers")
    parser.add_argument("--lr", default=0.001, type=float,
                        help="Learning rate to optimize the loss function")
    parser.add_argument("--decay_step", default=20000, type=float, help="Learning rate decay step")
    parser.add_argument("--lambda_val", default=0.5, type=float, help="Lambda factor for margin loss")
    parser.add_argument("--noise", default=0.3, type=float, help="dropout applied in input data")
    parser.add_argument("--Attention", default=True, type=bool, help="If use Attention module")
    parser.add_argument("--reg_scale", default=0.1, type=float, help="Regualar scale (reconstruction loss)")
    parser.add_argument("--coordinate", default=False, type=bool, help="If use Location record")
    parser.add_argument("--layer_depth", type=int, default=5, help="number of iterations of dynamic routing")
    parser.add_argument("--layer_width", type=int, default=2, help="number of iterations of dynamic routing")
    parser.add_argument("--num_graph_capsules", type=int, default=16, help="number of iterations of dynamic routing")
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print('device : ', device, flush=True)

    extn = '.gexf'
    class_labels_fname = args.dataset_dir + '.Labels'
    train_test_split_file = args.dataset_dir + '_train_test_split'

    gd = GraphDataset(input_dir=args.dataset_dir, extn=extn, class_label_fname=class_labels_fname)
    gd.print_status()

    with open(train_test_split_file, 'rb') as f:
        train_test_split_groups = pickle.load(f)

    test_accuracies = []
    for train_test_split_groups_idx, groups_dict in enumerate(train_test_split_groups):

        model = Model(args, gd.attri_len, gd.num_classes, gd.reconstruct_num, device).to(device)
        optimizer = optim.Adam(model.parameters(), args.lr)
        print(model)
        groups_dict = train_test_split_groups[train_test_split_groups_idx]
        gd.graphs_dataset_train = groups_dict['train']
        gd.graphs_dataset_valid = groups_dict['val']
        gd.graphs_dataset_test = groups_dict['test']

        max_val_acc = 0
        max_test_acc = 0
        max_eopch = 0
        for epoch in range(1, args.epochs+1):
            loss_accum = train(args, model, optimizer, gd.graphs_dataset_train, gd)

            train_acc = test(args, model, gd.graphs_dataset_train, gd, 'train')
            val_acc = test(args, model, gd.graphs_dataset_valid, gd, 'val')
            test_acc = test(args, model, gd.graphs_dataset_test, gd, 'test')

            if max_val_acc <= val_acc:
                max_val_acc = val_acc
                max_test_acc = test_acc
                max_eopch = epoch

            if epoch % 10 == 0:
                print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))
                print("accuracy train: %f val: %f test: %f" % (train_acc, val_acc, test_acc), flush=True)
                print('max val :', max_val_acc, 'test :', max_test_acc, 'epoch :', max_eopch)
                print('', flush=True)

        test_accuracies.append(max_test_acc)

    print(test_accuracies)
    print('avg :', sum(test_accuracies) / len(test_accuracies))


if __name__ == '__main__':
    main()
