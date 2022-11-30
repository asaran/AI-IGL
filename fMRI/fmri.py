import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from networks_nn import *

num_class = 3

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

def train_CB(args, CB_train_loader, reward, device, test_loader):
    optimzier = args.optimizer
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters())
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    epochs = args.epochs
    for epoch in range(epochs):
        for batch_idx, (x, a_idx, r) in enumerate(CB_train_loader):
            x, a_idx, r = x.to(device), a_idx.to(device), r.to(device).float()
            reward_optimizer.zero_grad()
            temp_idx = torch.arange(len(r)).to(device)
            estimated_rwds_output = reward(x)
            estimated_rwds = estimated_rwds_output[temp_idx, a_idx].reshape(r.shape)

            loss = 10 * F.mse_loss(estimated_rwds, r)
            loss.backward()
            reward_optimizer.step()
            if args.ifprint or True:
                if epoch % int(epochs / 10) == 0 and batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(CB_train_loader.dataset),
                        100. * batch_idx / len(CB_train_loader), loss.item()))
                    test(reward, test_loader, device)


def train_old_igl(args, train_loader, reward, decoder, device, test_loader):
    optimzier = args.optimizer
    adamlr = 1e-5
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters(), lr=adamlr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=adamlr)
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    epochs = args.epochs * 4

    for epoch in range(epochs):
        for batch_idx, (x, a, y, r) in enumerate(train_loader):
            x, a, y = x.to(device), a.to(device), y.to(device)
            reward_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            estimated_rwds = reward(x)
            decoded_rwds = decoder(y)

            temp_idx = torch.arange(len(decoded_rwds)).to(device)

            temp = estimated_rwds[temp_idx,a].reshape(decoded_rwds.shape)

            rwd_loss = torch.mean(temp) * torch.mean(decoded_rwds.detach()) - torch.mean(torch.mul(temp, decoded_rwds.detach()))
            decoder_loss = torch.mean(temp.detach()) * torch.mean(decoded_rwds) - torch.mean(torch.mul(temp.detach(), decoded_rwds))

            rwd_loss *= 100
            rwd_loss.backward()
            reward_optimizer.step()

            decoder_loss *= 100
            decoder_loss.backward()
            decoder_optimizer.step()

        # test if need to flip
        score = 0
        number = len(train_loader.dataset)
        score_phi = 0

        for _, (x, a, y, r) in enumerate(train_loader):
            x, a, y = x.to(device), a.to(device), y.to(device)
            estimated_rwds = reward(x)
            decoded_rwds = decoder(y)
            temp_idx = torch.arange(len(decoded_rwds)).to(device)
            temp = estimated_rwds[temp_idx,a].reshape(decoded_rwds.shape)
            score += torch.sum(temp).item()
            score_phi += torch.sum(decoded_rwds).item()

        if score / number > 0.5:
            for action in range(num_class):
                reward.flip(action)

        if score_phi / number > 0.5:
            decoder.flip(0)

        if epoch % int(epochs / 10) == 0 and (args.ifprint or True):
            test(reward, test_loader, device)


def train_sup(args, train_loader, test_loader, reward, device):
    optimzier = args.optimizer
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters())
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    epochs = args.epochs * 4

    for epoch in range(epochs):
        reward.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            reward_optimizer.zero_grad()
            output = reward.predict(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            reward_optimizer.step()


        if epoch % int(epochs / 10) == 0 and (args.ifprint or True):
            test(reward, test_loader, device)



def train_one_action(args, action, train_loader, reward, decoder, device):
    optimzier = args.optimizer
    adamlr = 1e-5
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters(), lr=adamlr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=adamlr)
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    action_loder = train_loader[action]

    epochs = args.epochs * 4

    for epoch in range(epochs):
        for batch_idx, (x, y, r) in enumerate(action_loder):
            x, y, r = x.to(device), y.to(device), r.to(device)
            reward_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            estimated_rwds = reward(x)[:,action]
            decoded_rwds = decoder(y)[:,action]

            if epoch < 2 * epochs / 3 + 1:
                rwd_loss = torch.mean(estimated_rwds) * torch.mean(decoded_rwds.detach()) - torch.mean(torch.mul(estimated_rwds, decoded_rwds.detach()))
                decoder_loss = torch.mean(estimated_rwds.detach()) * torch.mean(decoded_rwds) - torch.mean(torch.mul(estimated_rwds.detach(), decoded_rwds))

                decoder_loss *= 100
                decoder_loss.backward()
                decoder_optimizer.step()
            else:
                rwd_loss = F.mse_loss(estimated_rwds, decoded_rwds.detach())

            rwd_loss *= 100
            rwd_loss.backward()
            reward_optimizer.step()

            if epoch % int(epochs / 10) == 0 and batch_idx % args.log_interval == 0 and args.ifprint:
                print('Train Action {} Epoch: {} [{}/{} ({:.0f}%)]\tReward Obj: {:.6f}'.format(action,
                    epoch, batch_idx * len(x), len(action_loder.dataset),
                    100. * batch_idx / len(action_loder), -1 * rwd_loss.item()))

    # test if need to flip
    a_score = 0
    a_number = len(action_loder.dataset)

    for _, (x, y, r) in enumerate(action_loder):
        x, y, r = x.to(device), y.to(device), r.to(device)
        estimated_rwds = decoder(y)[:,action]
        a_score += torch.sum(estimated_rwds).item()

    if a_score / a_number > 0.5:
        reward.flip(action)


def test(reward, test_loader, device):
    reward.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = reward.predict(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# collect data w/ num_samples
def collect_igl_data(data_imagination,data_action,data_feedback,data_reward):
    data = [[] for _ in range(num_class)]
    length = data_imagination.shape[0]
    for i in tqdm(range(length)):
        x = data_imagination[i]
        a = int(data_action[i])
        y = data_feedback[i]
        r = data_reward[i]
        data[a].append([x,y,r])
    return data

def collect_test_data(data_imagination_test,data_context_test):
    data = []
    length = data_imagination_test.shape[0]
    for i in tqdm(range(length)):
        x = data_imagination_test[i]
        y = data_context_test[i]
        data.append([x,y])
    return data

def collect_test_data_sup(data_imagination_test,data_context_test):
    data = []
    length = int(data_imagination_test.shape[0] * 0.8)
    for i in tqdm(range(data_imagination_test.shape[0] - length)):
        x = data_imagination_test[i + length]
        y = data_context_test[i + length]
        data.append([x,y])
    return data

def collect_train_data_sup(data_imagination_test,data_context_test):
    data = []
    length = int(data_imagination_test.shape[0] * 0.8)
    for i in tqdm(range(length)):
        x = data_imagination_test[i]
        y = data_context_test[i]
        data.append([x,y])
    return data

# collect CB data w/ num_samples
def collect_CB_data(data_imagination,data_action,data_reward):
    data = []
    length = data_imagination.shape[0]
    for i in tqdm(range(length)):
        x = data_imagination[i]
        a = int(data_action[i])
        rwd = data_reward[i]
        data.append([x,a,rwd])
    return data

# collect old_IGL data w/ num_samples
def collect_old_igl_data(data_imagination,data_action,data_feedback,data_reward):
    data = []
    length = data_imagination.shape[0]
    for i in tqdm(range(length)):
        x = data_imagination[i]
        a_idx = int(data_action[i])
        y = data_feedback[i]
        rwd = data_reward[i]
        data.append([x,a_idx,y,rwd])
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IGL xCI MNIST batch model')
    parser.add_argument('--num_samples', type=int, default=20000, metavar='N',
                        help='how many samples for traning data (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=500 * 2, metavar='N',
                        help='how many epochs for training (default: 1000)')
    parser.add_argument('--evaluation_trials', type=int, default=64, metavar='N',
                        help='how many trails for evaluation (default: 64)')
    parser.add_argument('--lr', type=float, default=2e-2, metavar='LR',
                        help='learning rate (default: 2e-2)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                        help='momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', metavar='opt',
                        help='optimizer, adam or sgd, default is sgd')
    parser.add_argument('--cb', type=bool, default=False, metavar='CB',
                        help='if run CB')
    parser.add_argument('--igl', action='store_true', default=False, 
                        help='if run IGL')
    parser.add_argument('--old_igl', action='store_true', default=False, 
                        help='if run old IGL')
    parser.add_argument('--ifprint', type=bool, default=False, metavar='if print',
                        help='if print log')
    parser.add_argument('--noise_weight', type=float, default=0.01, metavar='N',
                        help='noise weight')
    parser.add_argument('--sup', type=bool, default=False, metavar='supervised learning',
                        help='if run supervised learning')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # load data
    noise_weight = str(args.noise_weight)

    data_imagination = np.load('data/data_imagination_combined_noise_level_'+noise_weight+'.npy')
    data_imagination = np.float32(data_imagination)
    data_action = np.load('data/data_action_combined_noise_level_'+noise_weight+'.npy')
    data_feedback = np.load('data/data_feedback_combined_noise_level_'+noise_weight+'.npy')
    data_feedback = np.float32(data_feedback)
    data_reward = np.load('data/data_reward_combined_noise_level_'+noise_weight+'.npy')
    data_imagination_test = np.load('data/data_imagination_test_combined_noise_level_'+noise_weight+'.npy')
    data_imagination_test = np.float32(data_imagination_test)
    data_context_test = np.load('data/data_context_test_combined_noise_level_'+noise_weight+'.npy').astype(int)

    train_kwargs = {'batch_size': args.batch_size,
                        'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size,
                        'shuffle': True}

    igl_data = collect_igl_data(data_imagination,data_action,data_feedback,data_reward)
    test_data = collect_test_data(data_imagination_test,data_context_test)

    test_loader = torch.utils.data.DataLoader(test_data,**test_kwargs)

    if args.igl:
        print('############################# running IGL #############################')

        # Train IGL Agent
        reward = reward_Net(x_size=64,a_size=num_class).to(device)
        # reward.init()
        decoder = decoder_Net(y_size=64,a_size=num_class).to(device)
        # decoder.init()

        train_loader = []
        for a in range(num_class):
            train_loader.append(torch.utils.data.DataLoader(igl_data[a], **train_kwargs))

        # Train IGL Agent
        test(reward, test_loader, device)
        for action in range(num_class):
            train_one_action(args, action, train_loader, reward, decoder, device)
            test(reward, test_loader, device)
        test(reward, test_loader, device)

    if args.old_igl:
        print('############################# running old IGL #############################')
        # old IGL Agent
        old_igl_reward = reward_Net(x_size=64,a_size=num_class).to(device)
        old_igl_decoder = decoder_Net(a_size=1,y_size=64).to(device)

        old_igl_data = collect_old_igl_data(data_imagination,data_action,data_feedback,data_reward)
        old_igl_train_loader = torch.utils.data.DataLoader(old_igl_data, **train_kwargs)

        # Train old IGL Agent
        test(old_igl_reward, test_loader, device)
        train_old_igl(args, old_igl_train_loader, old_igl_reward, old_igl_decoder, device, test_loader)
        test(old_igl_reward, test_loader, device)

    if args.sup:
        print('############################# running supervised learning #############################')
        # old IGL Agent
        sup_reward = reward_Net(x_size=64,a_size=num_class).to(device)
        test_data_sup = collect_test_data_sup(data_imagination_test,data_context_test)
        train_data_sup = collect_train_data_sup(data_imagination_test,data_context_test)
        test_loader_sup = torch.utils.data.DataLoader(test_data_sup,**test_kwargs)
        train_loader_sup = torch.utils.data.DataLoader(train_data_sup,**test_kwargs)

        # Train old IGL Agent
        test(sup_reward, test_loader_sup, device)
        train_sup(args, train_loader_sup, test_loader_sup, sup_reward, device)
        test(sup_reward, test_loader_sup, device)

if __name__ == '__main__':
    main()