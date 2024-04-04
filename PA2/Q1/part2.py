import os

import torch
from torch import tensor
import torchaudio
import librosa
from torchaudio.datasets import VoxCeleb1Verification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

#helper functions

#calculate eer
# def calculate_eer_percentage(true_labels, predictions):
#     cm = confusion_matrix(true_labels, predictions)
#     print(cm)
#     tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

#     far = fp / (fp + tn)  # False Acceptance Rate (FAR)
#     frr = fn / (fn + tp)  # False Rejection Rate (FRR)

#     eer = (far + frr) / 2
#     eer_percentage = eer * 100

#     return eer_percentage

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

    # assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    # model = init_model(model_name, checkpoint)

    # wav1, sr1 = sf.read(wav1)
    # wav2, sr2 = sf.read(wav2)

    # wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    # wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
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


class AudioDataset(Dataset):
    def __init__(self, length, num_samples, fixed_length):
        self.length = length
        self.num_samples = num_samples
        self.fixed_length = fixed_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random audio samples
        audio1 = torch.randn(self.fixed_length)
        audio2 = torch.randn(self.fixed_length)
        
        # Generate a random label
        label = torch.randint(0, 2, (1,), dtype=torch.long)
        
        return audio1, audio2, label


length = 1000  # Number of samples in the dataset
num_samples = 2  # Number of random audio samples to return per item
fixed_length = 44100  # Fixed length of each random audio sample


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


# dataset = AudioDataset(root="/DATA/arora8/SpeechUnderstanding/PA2/Q1/voxceleb1")
dataloader = DataLoader(dataset, batch_size=2)


model = init_model("wavlm_base_plus", checkpoint="/DATA/arora8/SpeechUnderstanding/PA2/Q1/checkpoints/wavelmbase+/WavLM-Base+.pt")
model.eval()

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