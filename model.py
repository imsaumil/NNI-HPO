# Importing required libraries
from __future__ import print_function
import argparse
import nni
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary

if __name__ == '__main__':

    # Creating arguments parser
    parser = argparse.ArgumentParser(description='ResNet light CIFAR-10 Example')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Setting the random seed
    torch.manual_seed(args.seed)

    # Using GPU, if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Device selected for training and testing
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device used: ", device, "\n")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Defining parameters to be tuned
    params = {
        'dropout_rate': 0.1,
        'lr': 0.001,
        'momentum': 0,
        "batch_size": 64
    }

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Fetching the next optimized hyperparameter
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    # Loading the training and testing data in dataloaders
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=params['batch_size'], shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=params['batch_size'], shuffle=False, drop_last=True, **kwargs)


    # Creating a ResNet block
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, downsample):
            super().__init__()
            if downsample:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                self.shortcut = nn.Sequential()

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, input):
            shortcut = self.shortcut(input)
            input = nn.ReLU()(self.bn1(self.conv1(input)))
            input = nn.ReLU()(self.bn2(self.conv2(input)))
            input = input + shortcut
            return nn.ReLU()(input)


    class ResNet(nn.Module):
        def __init__(self, in_channels, resblock, repeat, outputs=10):
            super().__init__()
            self.dropout = nn.Dropout(params['dropout_rate'])
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

            # Specifying custom filter numbers
            filters = [64, 64, 256]

            self.layer1 = nn.Sequential()
            self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
            for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d' % (i + 1,), resblock(filters[1], filters[1], downsample=False))

            self.layer2 = nn.Sequential()
            self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
            for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i + 1,), resblock(filters[2], filters[2], downsample=False))

            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(filters[2], outputs)

        def forward(self, input):
            input = self.layer0(input)
            input = self.layer1(input)
            input = self.layer2(input)
            input = self.gap(input)
            input = torch.flatten(input, start_dim=1)
            input = self.dropout(input)
            input = self.fc(input)
            return input

    # Defining the custom light ResNet model
    model = ResNet(1, ResBlock, [1, 1], outputs=10).to(device)

    # Specifying the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Specifying SGD optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=params["lr"], momentum=params["momentum"]
    )

    # Model training function
    def train(epoch=args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    # Model testing function
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        size = len(test_loader.dataset)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        correct /= size
        return correct


    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train()
        accuracy = test()
        nni.report_intermediate_result(accuracy * 100.)
    nni.report_final_result(accuracy)
