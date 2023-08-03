import torch.nn as nn

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResidualCNN, self).__init__()
        self.cnn1=nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2=nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1=nn.Dropout(dropout)
        #1 dropout ao inves de 2

    def forward(self,x):
        residual = x
        x=nn.functional.relu(x)
        x=self.dropout1(x)
        x=self.cnn1(x)
        x=nn.functional.relu(x)
        x=self.cnn2(x)
        x+=residual
        return x
class LSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size,dropout, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=rnn_dim,hidden_size=hidden_size,
                          batch_first=batch_first,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x=nn.functional.relu(x)
        x,_=self.lstm(x)
        x=self.dropout(x)
        return x
class SpeechRecognitionModel(nn.Module):
    #TENTATIVA 1 deus e bom
    def __init__(self,n_cnn_layers,n_rnn_layers,rnn_dim,n_class,stride=2, dropout=0.1):
        super(SpeechRecognitionModel,self).__init__()
        self.cnn = nn.Conv2d(1,32,3,stride=stride, padding=3//2)
        #n camadas residuais com 32 filtros cada
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32,32,kernel=3,stride=1,dropout=dropout)

            for _ in range(n_cnn_layers)
            ])
        self.fully_connected = nn.Linear(2048,rnn_dim)
        self.lstm_layers=nn.Sequential(*[
            #se der merda olha o batchfirst
            LSTM(rnn_dim=rnn_dim,hidden_size=rnn_dim,batch_first=True,dropout=dropout)
            for _ in range(n_rnn_layers)
        ])
        self.classifier=nn.Sequential(
            nn.Linear(rnn_dim,rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    def forward(self,x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes=x.size()
        x = x.view(sizes[0],sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1,2)
        x = self.fully_connected(x)
        x = self.lstm_layers(x)
        x = self.classifier(x)
        return x

