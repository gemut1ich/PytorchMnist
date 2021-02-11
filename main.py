import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # learn
        x = torch.flatten(x, 1)  # learn  压缩维度，可以指定开始压缩的维度和结束的维度
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # learn 好算，数值稳定 注意此时已经用了softmax，为之后用nll做铺垫
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # 告诉我的模型我在train这个模型
    for batch_idx, (data, target) in enumerate(
            train_loader):  # enumerate()里面的参数第一个是个迭代器，第二个参数是计数器，默认迭代从0开始，也可以设置成从第1个或者其他的开始
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # learn grad是默认累加的，所以在每轮需要清零，不累加的话可以玩出别的花
        output = model(data)
        loss = F.nll_loss(output, target)  # learn nll是negative log-likelihood，对garget对应的output做负log（偏差越大loss越大）
        loss.backward()
        optimizer.step() #梯度下降
        if batch_idx % args.log_interval == 0: #输出记录
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  # learn
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval() #告诉我的模型我在评估这个模型
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        #
        # print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%\n)".format(
        #     test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument("--epoch", type=int, default=14, help="number of batch to train (default:14)")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate (default:1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma(default:0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="how many batch to wait before logging train status")
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)  # 为保证结果稳定需要固定seed

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}  # kwargs全称是keyword arguements
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # Facebook自己算的mean和std https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
    ])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device) #model就是神经网络Net CNN
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) #用optimizer梯度下降

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
