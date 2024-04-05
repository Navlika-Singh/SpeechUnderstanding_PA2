#required imports
import os

import torch
from torch import tensor
import torchaudio
import numpy as np

import librosa

from torchaudio.datasets import VoxCeleb1Verification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.functional import pad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from sklearn import metrics


#helper functions

#calculate eer
def calculate_eer_percentage(labels, preds):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

#verification
import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]

def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification(model,  wav1, wav2, use_gpu=True, checkpoint=None):

    resample1 = Resample(orig_freq=16000, new_freq=16000)
    resample2 = Resample(orig_freq=16000, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    # if use_gpu:
    #     model = model.cuda()
    #     wav1 = wav1.cuda()
    #     wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return sim

#dataloader
class AudioDataset(Dataset):
    def __init__(self, root):

        self.root = root
        self.src1, self.src2, self.labels = self.read_data()
        self.read_hindidata()

    def read_data(self):
        src1 = []
        src2 = []
        labels = []

        file = open(self.root, 'r')
        for line in file:
            line = line.strip().split()
            src1.append(line[1])
            src2.append(line[2])
            labels.append(int(line[0]))
        
        return src1, src2, labels
    
    def read_hindidata(self):

        for i in range(len(self.src1)):
            
            split_path = self.src1[i].split('hindi')[1]
            split_path = split_path.split('/')
            
            new_path = os.path.join("test_known",split_path[2],split_path[4])
            new_path = os.path.join("/DATA/arora8/SpeechUnderstanding/PA2/Q1/kathbath",new_path)
            
            new_path = new_path.replace('.wav', '.m4a')
            
            self.src1[i] = new_path

        for i in range(len(self.src2)):
            
            split_path = self.src2[i].split('hindi')[1]
            split_path = split_path.split('/')
            
            new_path = os.path.join("test_known",split_path[2],split_path[4])
            new_path = os.path.join("/DATA/arora8/SpeechUnderstanding/PA2/Q1/kathbath",new_path)
            
            new_path = new_path.replace('.wav', '.m4a')
            
            self.src2[i] = new_path
    
    def __getitem__(self, idx):
        src1, src1_samplingrate = librosa.load(self.src1[idx])
        src2, src2_samplingrate = librosa.load(self.src2[idx])

        src1 = tensor(src1, dtype=torch.float32)
        src1 = src1.unsqueeze(0)
        src2 = tensor(src2, dtype=torch.float32)
        src2 = src2.unsqueeze(0)

        if src1.size(1) < 16000:
            src1 = pad(src1, (0, 16000 - src1.size(1)))
        else:
            src1 = src1[:, :16000]
        
        if src2.size(1) < 16000:
            src2 = pad(src2, (0, 16000 - src2.size(1)))
        else:
            src2 = src2[:, :16000]
        
        return src1[0], src2[0], self.labels[idx]

#initialise dataloader
dataset = AudioDataset(root="/DATA/arora8/SpeechUnderstanding/PA2/Q1/kathbath")
dataloader = DataLoader(dataset, batch_size=2)

#initialise model
model = init_model("wavlm_base_plus", checkpoint="/DATA/arora8/SpeechUnderstanding/PA2/Q1/checkpoints/wavelmbase+/WavLM-Base+.pt")
model.eval()

#calculate eer values
eer_batchwise = []

for i, data in enumerate(dataloader):

    src1, src2, label = data[0], data[1], data[2]

    src1 = src1
    src2 = src2

    similarity_scores = verification(model, src1, src2)
    print(similarity_scores)
    similarity_scores = (similarity_scores >= 0.0).float()
    predictions = similarity_scores

    eer_score = calculate_eer_percentage(label, predictions)
    print("[Batch {}] EER score: {}".format(i, eer_score))
    eer_batchwise.append(eer_score)

print("EER Score:", sum(eer_batchwise)/len(eer_batchwise))
