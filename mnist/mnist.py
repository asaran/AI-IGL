import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from networks import *

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

# MNIST Digit-ID Environment
class MNISTEnv(object):
    def __init__(self, x_size = 784, a_size = 10, y_size = 784):
        self.mnist = MNIST('./data', train=True, download=True,
                        transform=transform)
        self.mnist.data = self.mnist.data / 255.0
        self.x_size = x_size
        self.y_size = y_size
        self.a_size = a_size
        # Cache the indecies of all 0 and 1 digit images
        self.labeled_images = [(self.mnist.targets == i).nonzero().view(-1) for i in range(10)]
        self.x = None
        self.label = None
        self.reward = None

    def observe(self):
        # Sample a random digit to provide as the context
        self.id = torch.randint(0, self.mnist.data.shape[0], [1])
        self.x = self.mnist.data[self.id]
        self.label = self.mnist.targets[self.id].item()
        self.done = False
        return self.x.view(1, -1)

    # if r = 1, return img w/ label = (a + 1) % 10, otherwise, label = (a - 1) % 10
    # return (y, reward)
    def step(self, action):
        self.reward = int(self.label == action)
        # y_label = (action + 6 * self.reward - 3) % 10
        y_label = self.reward
        y_img_idx = self.labeled_images[y_label][torch.randint(0, len(self.labeled_images[y_label]), [1])]
        return self.mnist.data[y_img_idx].view(1, -1), self.reward

# collect data w/ num_samples
def collect_data(env, num_samples):
    data = [[] for _ in range(10)]
    for _ in tqdm(range(num_samples)):
        x = env.observe()
        a = torch.randint(low=0, high=10, size=(1,)).item()
        y, rwd = env.step(a)
        data[a].append([x,y,rwd])
    return data

# collect CB data w/ num_samples
def collect_CB_data(env, num_samples):
    data = []
    for _ in tqdm(range(num_samples)):
        x = env.observe()
        a = torch.randint(low=0, high=env.a_size, size=(1,)).item()
        # a = env.openml.targets[a_idx]
        _, rwd = env.step(a)
        data.append([x,a,rwd])
    return data

# collect old_IGL data w/ num_samples
def collect_old_igl_data(env, num_samples):
    data = []
    for _ in tqdm(range(num_samples)):
        x = env.observe()
        a_idx = torch.randint(low=0, high=env.a_size, size=(1,)).item()
        # a = env.openml.targets[a_idx]
        y, rwd = env.step(a_idx)
        data.append([x,a_idx,y,rwd])
    return data

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
            if args.ifprint:
                if epoch % int(epochs / 10) == 0 and batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(CB_train_loader.dataset),
                        100. * batch_idx / len(CB_train_loader), loss.item()))
                    test(reward, test_loader, device)


def train_old_igl(args, train_loader, reward, decoder, device, test_loader):
    optimzier = args.optimizer
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters())
        decoder_optimizer = optim.Adam(decoder.parameters())
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    epochs = args.epochs

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

            # if epoch % int(epochs / 10) == 0 and batch_idx % args.log_interval == 0 and args.ifprint:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\Reward Obj: {:.6f}\tDecoder Obj: {:.6f}'.format(
            #         epoch, batch_idx * len(x), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), -1 * rwd_loss.item(), -1 * decoder_loss.item()))

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
            for action in range(10):
                reward.flip(action)

        if score_phi / number > 0.5:
            decoder.flip(0)

        if epoch % int(epochs / 10) == 0 and args.ifprint:
            test(reward, test_loader, device)




def train_one_action(args, action, train_loader, reward, decoder, device):
    optimzier = args.optimizer
    if optimzier == 'adam':
        reward_optimizer = optim.Adam(reward.parameters())
        decoder_optimizer = optim.Adam(decoder.parameters())
    elif optimzier == 'sgd':
        reward_optimizer = optim.SGD(reward.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    action_loder = train_loader[action]

    epochs = args.epochs

    for epoch in range(epochs):
        for batch_idx, (x, y, r) in enumerate(action_loder):
            x, y = x.to(device), y.to(device)
            reward_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            estimated_rwds = reward(x)[:,action]
            decoded_rwds = decoder(y)[:,action]

            if epoch < epochs / 2:
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
        x, y = x.to(device), y.to(device)
        estimated_rwds = decoder(y)[:,action]
        a_score += torch.sum(estimated_rwds).item()

    if a_score / a_number > 0.5:
        reward.flip(action)


def test(reward, test_loader, device):
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
    parser.add_argument('--cb', action='store_true', default=False,
                        help='if run CB')
    parser.add_argument('--igl', action='store_true', default=False, 
                        help='if run IGL')
    parser.add_argument('--old_igl', action='store_true', default=False, 
                        help='if run old IGL')
    parser.add_argument('--ifprint', type=bool, default=False, metavar='if print',
                        help='if print log')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    env = MNISTEnv()

    train_kwargs = {'batch_size': args.batch_size,
                        'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size,
                        'shuffle': True}

    test_data = datasets.MNIST('./data', train=False, download=True,
                        transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data,**test_kwargs)

    if args.igl:
        print('############################# running IGL #############################')

        # generate data
        data = collect_data(env, args.num_samples)

        # Train IGL Agent
        reward = reward_Net().to(device)
        decoder = decoder_Net().to(device)

        train_loader = []
        for a in range(10):
            train_loader.append(torch.utils.data.DataLoader(data[a], **train_kwargs))

        # Train IGL Agent
        test(reward, test_loader, device)
        for action in range(10):
            train_one_action(args, action, train_loader, reward, decoder, device)
            test(reward, test_loader, device)
        test(reward, test_loader, device)

    if args.cb:
        print('############################# running CB #############################')
        # CB Agent
        CB_reward = reward_Net(x_size=env.x_size,a_size=env.a_size).to(device)
        CB_data = collect_CB_data(env, args.num_samples)
        CB_train_loader = torch.utils.data.DataLoader(CB_data, **train_kwargs)

        # Train CB Agent
        train_CB(args, CB_train_loader, CB_reward, device, test_loader)
        test(CB_reward, test_loader, device)

    if args.old_igl:
        print('############################# running old IGL #############################')
        # old IGL Agent
        old_igl_reward = reward_Net(x_size=env.x_size,a_size=env.a_size).to(device)
        old_igl_data = collect_old_igl_data(env, args.num_samples)
        old_igl_decoder = decoder_Net(a_size=1,y_size=env.y_size).to(device)
        old_igl_train_loader = torch.utils.data.DataLoader(old_igl_data, **train_kwargs)

        # Train old IGL Agent
        test(old_igl_reward, test_loader, device)
        train_old_igl(args, old_igl_train_loader, old_igl_reward, old_igl_decoder, device, test_loader)
        test(old_igl_reward, test_loader, device)

if __name__ == '__main__':
    main()
