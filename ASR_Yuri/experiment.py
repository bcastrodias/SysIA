import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchaudio
from dataprocess import data_processing
from model import SpeechRecognitionModel
from train import train


hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,#ja tentei 5
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.3,
        "learning_rate": 5e-4,#ja tentei 5e-3
        "batch_size": 10,
        "epochs": 10
    }
torch.manual_seed(420)
device=torch.device("cuda")

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

train_loader=data.DataLoader(dataset=train_dataset,
                             batch_size=hparams['batch_size'],
                             shuffle=True,
                             collate_fn=lambda x: data_processing(x, 'train')
                             )
model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['stride'], hparams['dropout']
        ).to(device)
optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
criterion = nn.CTCLoss(blank=28).to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                          steps_per_epoch=int(len(train_loader)),
                                          epochs=hparams['epochs'],
                                          anneal_strategy='linear')
for epoch in range(1, hparams["epochs"] + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
torch.save(model.state_dict(),"trainedmodel/asr_yuri.pt")