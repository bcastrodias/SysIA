
import torch.nn as nn


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data, in enumerate(train_loader):
        spectograms, labels, input_length, label_length = _data
        spectograms, labels, = spectograms.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(spectograms)
        output = nn.functional.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        #Transpose?
        loss=criterion(output, labels, input_length, label_length)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(spectograms), data_len,
                            100. * batch_idx / len(train_loader), loss.item()))


