import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


train_losses = {}
train_acc = {}
test_losses = {}
test_acc = {}


def train(model, device, train_loader, optim, epoch):
    model.train()
    train_loss = 0
    epoch_acc = 0
    correct = 0
    processed = 0
    pbar = tqdm(train_loader)
    for batch, (data, targets) in enumerate(pbar):
        optim.zero_grad()
        if device == torch.device("cuda"):
            data, target = data.to(device), targets.to(device)
        outputs = model(data)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        epoch_acc += ((preds == target).sum() / len(target)) * 100
        correct += (preds == target).sum().item()
        processed += len(target)
        pbar.set_description(
            f"Loss = {loss.item()},batch_idx = {batch}, accuracy = {100* correct/processed}"
        )

    train_losses[epoch] = train_loss
    train_acc[epoch] = correct / processed * 100


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    epoch_acc = 0
    correct = 0
    processed = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            if device == torch.device("cuda"):
                data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.nll_loss(outputs, target, reduction="sum").item()
            test_loss += loss
            pred = torch.argmax(outputs, dim=1)

            # epoch_acc += (((pred == target).sum() / len(target)) * 100).item()
            # epoch_acc += (pred == target).sum().item() / len(target)
            correct += (pred == target).sum().item()
            processed += len(target)
            # pbar.set_description(f"Test  batch for batch {batch}, loss =  {loss.item()},{correct}/{len(target)}, accuracy = {batch_acc}" )
    # test_loss /= len(test_loader.dataset)
    # print("The epoch_acc is ", epoch_acc)
    test_losses[epoch] = test_loss / len(test_loader.dataset)
    test_acc[epoch] = correct / processed * 100
    print(
        "\n Test set Avg Loss = {:.4f}, Accuracy {}/{},({:.2f}%)\n".format(
            test_loss / len(test_loader.dataset),
            correct,
            processed,
            100 * (correct / processed),
        )
    )
