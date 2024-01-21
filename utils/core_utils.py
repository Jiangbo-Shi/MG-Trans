import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss
from scipy.spatial.distance import pdist, squareform
from models.model_utils import calculate_MI


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss().cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'MG_Trans':
        from models.model_MG_Trans import MG_Trans_main, CONFIGS
        config = CONFIGS[args.model_type]
        config.select_ratio = args.select_ratio
        config.transformer.num_layers = args.layer_num
        config.transformer.num_heads = args.head_num
        model_dict = {'config': config, 'img_size': args.img_size, 'num_classes':args.n_classes, 'zero_head':True, 'smoothing_value':args.smoothing_value}
        model = MG_Trans_main(**model_dict)
        if(args.pretrained_dir != None):
            model.load_from(np.load(args.pretrained_dir))

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping and args.model_type == 'transFG':
        early_stopping = EarlyStopping(patience=20, stop_epoch=80, verbose=True)
    elif args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping: 
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(args.mode, model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    x1_tmp = torch.zeros((1, 192)).to(device)
    x2_tmp = torch.zeros((1, 192)).to(device)
    z1_tmp = torch.zeros((1, 192)).to(device)
    z2_tmp = torch.zeros((1, 192)).to(device)

    print('\n')
    for batch_idx, (data_s, coord_s, data_l, coords_l, label, attn_mask) in enumerate(loader):

        data_s, coord_s, data_l, coords_l, label, attn_mask = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device), attn_mask.to(device)
        logits, _, Y_hat, _, loss, x1, x2, z1, z2, out = model(data_s, coord_s, data_l, coords_l, attn_mask, label)

        if(args.model_type == 'MG_Trans'):
            if(int((batch_idx+1)/32) == 0):
                x1_tmp = torch.cat((x1_tmp, x1.detach()), dim=0)
                x2_tmp = torch.cat((x2_tmp, x2.detach()), dim=0)
                z1_tmp = torch.cat((z1_tmp, z1.detach()), dim=0)
                z2_tmp = torch.cat((z2_tmp, z2.detach()), dim=0)

            else:
                x1_tmp = torch.from_numpy(np.delete(x1_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
                x2_tmp = torch.from_numpy(np.delete(x2_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
                z1_tmp = torch.from_numpy(np.delete(z1_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)
                z2_tmp = torch.from_numpy(np.delete(z2_tmp.cpu().detach().numpy(), 0, axis=0)).to(device)

                x1_tmp = torch.cat((x1_tmp, x1), dim=0)
                x2_tmp = torch.cat((x2_tmp, x2), dim=0)
                z1_tmp = torch.cat((z1_tmp, z1), dim=0)
                z2_tmp = torch.cat((z2_tmp, z2), dim=0)
                with torch.no_grad():
                    x1_numpy = x1_tmp.cpu().detach().numpy()
                    k_x1 = squareform(pdist(x1_numpy, 'euclidean'))
                    sigma_x1 = np.mean(np.mean(np.sort(k_x1[:, :10], 1)))

                    x2_numpy = x2_tmp.cpu().detach().numpy()
                    k_x2 = squareform(pdist(x2_numpy, 'euclidean'))
                    sigma_x2 = np.mean(np.mean(np.sort(k_x2[:, :10], 1)))

                    Z1_numpy = z1_tmp.cpu().detach().numpy()
                    Z2_numpy = z2_tmp.cpu().detach().numpy()
                    k_z1 = squareform(pdist(Z1_numpy, 'euclidean'))
                    k_z2 = squareform(pdist(Z2_numpy, 'euclidean'))
                    sigma_z1 = np.mean(np.mean(np.sort(k_z1[:, :10], 1)))
                    sigma_z2 = np.mean(np.mean(np.sort(k_z2[:, :10], 1)))

                IX1Z1 = calculate_MI(x1_tmp, z1_tmp, s_x=sigma_x1 ** 2, s_y=sigma_z1 ** 2)
                IX2Z2 = calculate_MI(x2_tmp, z2_tmp, s_x=sigma_x2 ** 2, s_y=sigma_z2 ** 2)
                beta1 = 0.0001
                beta2 = 0.0001
                MIB_loss = beta1 * IX1Z1 + beta2 * IX2Z2
                loss = loss + MIB_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)

        loss_value = loss.item()
        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coords_l, label, attn_mask) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label, attn_mask = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True), attn_mask.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, loss, _, _, _, _, _ = model(data_s, coord_s, data_l, coords_l, attn_mask, label)

            acc_logger.log(Y_hat, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

    val_error /= len(loader)
    val_loss /= len(loader)
    val_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(mode, model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}


    for batch_idx, (data_s, coord_s, data_l, coords_l, label, attn_mask) in enumerate(loader):
        data_s, coord_s, data_l, coords_l, label, attn_mask = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device), attn_mask.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, loss, _, _, _, _, _ = model(data_s, coord_s, data_l, coords_l, attn_mask, label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        all_pred.append(Y_hat.cpu())
        all_label.append(label.cpu())

    test_error /= len(loader)
    test_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, test_f1

