import copy

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MobileNet


# Train Network
def train(train_loader, model, criterion, optimizer):
    # set model to train mode
    model.train()
    # train loop
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if available
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # optimization
            optimizer.step()

            # print loss
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()
    return


# Check accuracy on training & test to see how good our model
def eval(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    model.eval()

    with tqdm(total=len(loader)) as t:
        with torch.no_grad():
            for x, y in loader:
                # Get data to cuda if available
                x = x.to(device=device)
                y = y.to(device=device)
                # Forward
                scores = model(x)
                # Loss
                loss = criterion(scores, y)
                # print loss
                t.set_postfix(loss='{:05.3f}'.format(loss.item()))
                t.update()
                # evaluate correct predictions
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

    return num_correct/num_samples


if __name__ == '__main__':
    # Hyperparameters
    in_channels = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3
    pretrained = False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Mnist Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Model
    model = MobileNet(in_channels=in_channels, out_cls=num_classes, pretrained=pretrained).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        num_correct = eval(loader=test_loader, model=model, criterion=criterion)
        print(f"Accuracy on test set: {num_correct * 100:.2f}")
        if num_correct > best_acc:
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optim_dict': optimizer.state_dict()
                     }
            torch.save(state, 'best_chk.pt')

    print("Accuracies on train and test with final model")
    print(f"Accuracy on training set: {eval(loader=test_loader, model=model, criterion=criterion) * 100:.2f}")
    print(f"Accuracy on test set: {eval(loader=test_loader, model=model, criterion=criterion) * 100:.2f}")
