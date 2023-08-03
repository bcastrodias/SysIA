import torchaudio
import torch
import torch.nn as nn


char_map_str = """
 ' 0
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 """
class TextTransform:
    #classe pra ligar letras a ints
    def __init__(self,char_map_str=char_map_str):
        char_map_str=char_map_str
        self.char_map = {}
        self.index_map = {}
        for lin in char_map_str.strip().split('\n'):
            ch,index = lin.split()
            self.char_map[ch]=int(index)
            self.index_map[int(index)]=ch
        self.index_map[1]=' '
        self.char_map[' ']=1
    def text_to_int(self,text):
        #transformar texto em sequencia de inteiros
        sequencia=[]
        for i in text:
            sequencia.append(self.char_map[i])
        return sequencia
    def int_to_txt(self,seq):
        #caminho contrario
        text=""
        for i in seq:
            text=text+self.index_map[i]
        return text
#preprocessar dados
train_audio_transforms=torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
valid_audio_transforms=torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
text_transformer=TextTransform()
def data_processing(data, data_type="train"):
    spectograms=[]
    labels=[]
    input_length=[]
    label_length=[]
    for(waveform, _, utterance, _, _, _) in data:
        if data_type == "train":
            spec=train_audio_transforms(waveform).squeeze(0).transpose(0,1)
        else:
            spec=valid_audio_transforms(waveform).squeeze(0).transpose(0,1)
        spectograms.append(spec)
        label=torch.Tensor(text_transformer.text_to_int(utterance.lower()))
        labels.append(label)
        input_length.append(spec.shape[0]//2)
        label_length.append(len(label))
    spectograms=nn.utils.rnn.pad_sequence(spectograms, batch_first=True).unsqueeze(1).transpose(2,3)
    labels=nn.utils.rnn.pad_sequence(labels,batch_first=True)
    return spectograms, labels, input_length, label_length

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transformer.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transformer.int_to_text(decode))
    return decodes, targets


