import os

import librosa
import torch
from torch import tensor
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import SignalDistortionRatio

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# class AudioDataset(Dataset):
#     def __init__(self, root):

#         self.root = root
#         self.src1, self.src2, self.labels = self.read_data()

#     def read_data(self):
#         src1 = []
#         src2 = []
#         labels = []

#         datafile = open(self.root, 'r')
#         for dataline in datafile:
#             dataline = dataline.strip().split()
#             src1.append(dataline[1])
#             src2.append(dataline[2])
#             labels.append(dataline[0])
        
#         return src1, src2, labels
    
#     def __getitem__(self, idx):
#         src1, src1_samplingrate = torchaudio.load(os.path.join("/DATA/arora8/SpeechUnderstanding/PA2/Q1/voxceleb1/wav/", self.src1[idx]))
#         src2, src2_samplingrate = torchaudio.load(os.path.join("/DATA/arora8/SpeechUnderstanding/PA2/Q1/voxceleb1/wav/", self.src2[idx]))

#         if src1.size(1) < 16000:
#             src1 = pad(src1, (0, 16000 - src1.size(1)))
#         elif:
#             src1 = src1[:, :16000]
        
#         if src2.size(1) < 16000:
#             src2 = pad(src2, (0, 16000 - src2.size(1)))
#         elif:
#             src2 = src2[:, :16000]
        
#         return src1[0], src2[0], self.labels[idx]


class AudioDataset(Dataset):
    def __init__(self, root):

        self.root = root
        self.mixed_file, self.src1, self.src2 = self.read_data()
    
    def read_data(self):
        mixed_file = []
        src1 = []
        src2 = []

        mixed_sub_dir_path = os.path.join(self.root, "mix_clean")
        s1_sub_dir_path = os.path.join(self.root, "s1")
        s2_sub_dir_path = os.path.join(self.root, "s2")
        for file_name in os.listdir(mixed_sub_dir_path):
            mixed_file.append(os.path.join(mixed_sub_dir_path, file_name))
            src1.append(os.path.join(s1_sub_dir_path, file_name))
            src2.append(os.path.join(s2_sub_dir_path, file_name))
        
        return mixed_file, src1, src2

    def __len__(self):
        return len(self.mixed_file)
    
    def __getitem__(self, idx):

        mixed, mixed_samplingrate = librosa.load(self.mixed_file[idx])
        src1, src1_samplingrate = librosa.load(self.src1[idx])
        src2, src2_samplingrate = librosa.load(self.src2[idx])

        mixed = tensor(mixed, dtype=torch.float32)
        mixed = mixed.unsqueeze(0)
        src1 = tensor(src1, dtype=torch.float32)
        src1 = src1.unsqueeze(0)
        src2 = tensor(src2, dtype=torch.float32)
        src2 = src2.unsqueeze(0)

        if mixed.size(1) < 16000:
            mixed = pad(mixed, (0, 16000 - mixed.size(1)))
        else:
            mixed = mixed[:, :16000]

        if src1.size(1) < 16000:
            src1 = pad(src1, (0, 16000 - src1.size(1)))
        else:
            src1 = src1[:, :16000]
        
        if src2.size(1) < 16000:
            src2 = pad(src2, (0, 16000 - src2.size(1)))
        else:
            src2 = src2[:, :16000]
        
        return mixed[0], src1[0], src2[0]


dataset = AudioDataset("/DATA/arora8/SpeechUnderstanding/PA2/Q2/LibriMix/storage_dir/Libri2Mix/wav16k/max/test")
dataloader = DataLoader(dataset, batch_size=2)


model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

SISNR = ScaleInvariantSignalNoiseRatio()
SISDR = SignalDistortionRatio()

for mixed, src1, src2 in dataloader:
    print("Inside loop")
    est_sources = model.separate_batch(mixed) 
    # print(est_sources[:,:,0], est_sources[:,:,1])

    SISNR_og = SISNR(est_sources[:,:,0], mixed) + SISNR(est_sources[:,:,1], mixed)
    SISDR_og = SISDR(est_sources[:,:,0], mixed) + SISDR(est_sources[:,:,1], mixed)

    SISNR_i = SISNR(est_sources[:,:,0], src1) + SISNR(est_sources[:,:,1], src2)
    SISDR_i = SISDR(est_sources[:,:,0], src1) + SISDR(est_sources[:,:,1], src2)

    SISNR_i = SISNR_og - SISNR_i
    SISDR_i = SISDR_og - SISDR_i

    SISNR_i = SISNR_i.mean().item()/2
    SISDR_i = SISDR_i.mean().item()/2

    print(SISNR_i, SISDR_i)