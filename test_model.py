import pandas as pd
import torch
import argparse
import utils
import scipy.io as sio
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import xai
from efficient_kan import KAN
import torch.optim as optim
from tqdm import tqdm
from matplotlib.colors import Normalize
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--fold', type=int, default=1, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rage')#5e-2
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='epoch')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--step_size', type=int, default=50, help='step size')
parser.add_argument('--gamma', type=float, default=0.2, help='gamma')
args = parser.parse_args()



def train(X_train, Y_train, model, optimizer, criterion):
    loss_AM = utils.AvgrageMeter()
    acc_AM = utils.AvgrageMeter()
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True)#.to(device)
        target = Variable(target, requires_grad=True)
        optimizer.zero_grad()
        logits = model(input)
        #logits = logits.cpu()
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        accuracy = utils.accuracy(logits, target)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
    return loss_AM.avg, acc_AM.avg


def infer(X_vaild, Y_vaild, model, criterion):
    loss_AM = utils.AvgrageMeter()
    acc_AM = utils.AvgrageMeter()
    all_predictions = []
    all_targets = []
    model.eval()

    for step, (input, target) in enumerate(zip(X_vaild, Y_vaild)):
        with torch.no_grad():
            input = Variable(input)
        with torch.no_grad():
            target = Variable(target)

        logits = model(input)
        probabilities = torch.softmax(logits, dim=1)
        # 预测类别
        predictions = torch.argmax(probabilities, dim=1)
        loss = criterion(logits, target.long())
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        accuracy = utils.accuracy(logits, target)
        sen, spe = utils.sensitivity(logits, target)
        n = input.size(0)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    f1_score = metrics.f1_score(all_targets, all_predictions, average='macro')
    recall = metrics.recall_score(all_targets, all_predictions, average='macro')
    precision = metrics.precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    balanced_accuracy = metrics.balanced_accuracy_score(all_targets, all_predictions)
    auc_score = metrics.roc_auc_score(all_targets, probabilities[:, 1].detach().numpy())

    return loss_AM.avg, acc_AM.avg, sen, spe, f1_score, recall, precision, balanced_accuracy, auc_score


def infer2(X_vaild, Y_vaild, model, criterion):
    loss_AM = utils.AvgrageMeter()
    acc_AM = utils.AvgrageMeter()
    all_predictions = []
    all_targets = []
    model.eval()
    ln = nn.LayerNorm(normalized_shape=[90, 90], elementwise_affine=False)
    X_vaild = get_batch(ln(torch.tensor(X_vaild)).view(-1, 1, 90, 90).type(torch.FloatTensor))
    Y_vaild = get_batch(Y_vaild)
    for step, (input, target) in enumerate(zip(X_vaild, Y_vaild)):
        with torch.no_grad():
            input = Variable(input)
        with torch.no_grad():
            target = Variable(target)

        logits = model(input)
        probabilities = torch.softmax(logits, dim=1)
        # 预测类别
        predictions = torch.argmax(probabilities, dim=1)
        loss = criterion(logits, target.long())
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        accuracy = utils.accuracy(logits, target)
        sen, spe = utils.sensitivity(logits, target)
        n = input.size(0)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    f1_score = metrics.f1_score(all_targets, all_predictions, average='macro')
    recall = metrics.recall_score(all_targets, all_predictions, average='macro')
    precision = metrics.precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    balanced_accuracy = metrics.balanced_accuracy_score(all_targets, all_predictions)
    auc_score = metrics.roc_auc_score(all_targets, probabilities[:, 1].detach().numpy())

    return loss_AM.avg, acc_AM.avg, sen, spe, f1_score, recall, precision, balanced_accuracy, auc_score

def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0)
    return X_output


def testKAN():
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    sum_loss = 0
    sum_acc = 0
    sum_sen = 0
    sum_spe = 0
    fold_list = []
    load_data = sio.loadmat('E:\Brain\Convolutional_Architecture\data/ALLASD1_aal.mat')
    template_matrix = np.load('E:\Brain\Convolutional_Architecture\save_csv/sample_arch_90_find.npy',
                              allow_pickle=True)
    matrixAuto=[85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72,
                60, 40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30, 5, 34, 21, 68, 57, 88, 1, 83, 75, 14, 84, 39, 11, 70,
                76, 16, 12, 50, 77, 24, 18, 27, 63, 29, 20, 64, 35, 48, 74, 33, 47, 51, 4, 17, 2, 23, 61, 26, 10,
                71, 36, 58, 67, 7, 73, 37, 0, 22, 49, 38, 25, 41, 13,]
    matrixAuto4 =[85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72, 60, 40, 9, 87, 81,]
    matrixAuto5 =[85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72, 60,
                  40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30, 5, 34, 21, 68, 57, 88,]#Z-score<0
    matrixAuto6 =[85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72, 60,
                  40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30, 5, 34, 21, 68, 57, 88,
                  1, 83, 75, 14, 84, 39, 11, 70, 76, 16, 12, 50, 77, 24, 18, 27, 63, 29, 20, 64, 35, 48,]#Min-max normalization <0.5
    matrixAuto7 = [85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55,]
    matrixAuto8 = [85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72, 60,
                   40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30, 5, 34, 21, 68, 57, 88,
                   1, 83, 75, 14, 84, ]#Min-max normalization<0.46
    matrixAuto9 = [85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32, 59, 42, 8, 55, 44, 31, 72,
                60, 40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30, 5, 34, 21, 68, 57, 88, 1, 83, 75, 14, 84, 39, 11, 70,
                76, 16, 12, 50, 77, 24, 18, 27, 63, 29, 20, 64, 35, 48, 74, 33, 47, 51, 4, 17, 2, 23, 61, 26, 10,
                71, 36, 58,]#by Z-score <1
    acc=[]
    f1_score=[]
    recall=[]
    precision=[]
    balanced_accuracy=[]
    auc_score=[]
    for i in range(90):
        anum = i+1
        ln = nn.LayerNorm(normalized_shape=[anum, anum], elementwise_affine=False)
        matrix30 = matrixAuto[:anum]
        X_train = load_data['net_train']
        X_test = load_data['net_valid']
        #X_test = load_data['net_test']
        X_train = [matrix[np.ix_(matrix30, matrix30)] for matrix in X_train]
        X_test = [matrix[np.ix_(matrix30, matrix30)] for matrix in X_test]
        X_train = get_batch(ln(torch.tensor(X_train)).view(-1, 1, anum, anum).type(torch.FloatTensor))  # 90
        X_test = get_batch(ln(torch.tensor(X_test)).view(-1, 1, anum, anum).type(torch.FloatTensor))
        Y_train = get_batch(load_data['phenotype_train'][:, 2])
        Y_test = get_batch(load_data['phenotype_valid'][:, 2])
        #Y_test = get_batch(load_data['phenotype_test'][:, 2])
        trainloader = X_train
        valloader = X_test
        # Define model
        model = KAN([anum * anum,256,128,64,32,16,2])#90,512,256,128,64,32,16,8,4,2;30,512,256,128,64,32,16,2
        #model.plot()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        # Define loss
        criterion = nn.CrossEntropyLoss()

        for epoch in range(50):
            # Train
            model.train()
            with tqdm(zip(X_train, Y_train)) as pbar:
                for i, (images,labels) in enumerate(pbar):
                    images = images.view(-1, anum * anum).to(device)
                    optimizer.zero_grad()
                    output = model(images)
                    labels = labels.long()
                    loss = criterion(output, labels.to(device))
                    loss.backward()
                    optimizer.step()
                    accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
            model.eval()
            val_loss = 0
            val_accuracy = 0
            all_predictions = []
            all_targets = []
            with torch.no_grad():
                for images, labels in zip(valloader,Y_test):
                    images = images.view(-1, anum * anum).to(device)
                    #print(images.shape)
                    output = model(images)
                    probabilities = torch.softmax(output, dim=1)
                    # 预测类别
                    predictions = torch.argmax(probabilities, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    labels = labels.long()
                    val_loss += criterion(output, labels.to(device)).item()
                    val_accuracy += (
                        (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                    )

            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            val_loss /= len(valloader)
            val_accuracy /= len(valloader)
            val_f1_score = metrics.f1_score(all_targets, all_predictions, average='macro')
            val_recall = metrics.recall_score(all_targets, all_predictions, average='macro')
            val_precision = metrics.precision_score(all_targets, all_predictions, average='macro', zero_division=0)
            val_balanced_accuracy = metrics.balanced_accuracy_score(all_targets, all_predictions)
            val_auc_score = metrics.roc_auc_score(all_targets, probabilities[:, 1].cpu().detach().numpy())
            # Update learning rate
            scheduler.step()

            #print(
            #    f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
            #)
        print(anum)
        acc.append(val_accuracy)
        f1_score.append(val_f1_score)
        recall.append(val_recall)
        precision.append(val_precision)
        balanced_accuracy.append(val_balanced_accuracy)
        auc_score.append(val_auc_score)
    print(acc)
    print(f1_score)
    print(recall)
    print(precision)
    print(balanced_accuracy)
    print(auc_score)

    import datetime

    mystring = datetime.datetime.now().strftime("%m-%d-%H-%M-KAN")

    filename_pt = mystring + "_model.pt"
    #torch.save(model, filename_pt)
def infer3(x,y,model,criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anum = 90
    val_loss = 0
    val_accuracy = 0
    ln = nn.LayerNorm(normalized_shape=[anum, anum], elementwise_affine=False)
    x = get_batch(ln(torch.tensor(x)).view(-1, 1, 90, 90).type(torch.FloatTensor))
    y = get_batch(y)
    for images, labels in zip(x, y):
        images = images.view(-1, anum * anum).to(device)
        output = model(images)
        labels = labels.long()
        val_loss += criterion(output, labels.to(device)).item()
        val_accuracy += (
            (output.argmax(dim=1) == labels.to(device)).float().mean().item()
        )
    val_loss /= len(x)
    val_accuracy /= len(x)
    return val_accuracy, val_loss
def get_avg(x,y):
    model = torch.load('05-11-11-32-KAN-68_model.pt')
    criterion = nn.CrossEntropyLoss()
    x = x.to_numpy()
    x = x.reshape((84, 90, 90))
    return infer3(x, y, model, criterion)[1]
def testXAI():
    model = torch.load('05-11-11-32-KAN-68_model.pt')
    ln = nn.LayerNorm(normalized_shape=[90, 90], elementwise_affine=False)
    load_data = sio.loadmat('E:\Brain\Convolutional_Architecture\data/ALLASD1_aal.mat')
    x = load_data['net_test']
    print(x.shape)
    y = load_data['phenotype_test'][:, 2]
    print(y.shape)
    criterion = nn.CrossEntropyLoss()
    x = x.reshape(x.shape[0], -1)
    x = pd.DataFrame(x)
    imp = xai.feature_importance(x,y,get_avg)
    imp.to_csv('feature_importance.csv')
    imp.head()
if __name__ == "__main__":
    testKAN()
    #testXAI()
