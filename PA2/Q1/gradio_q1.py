#required imports

import gradio as gr
import numpy as np
import torch

from speechbrain.pretrained import SepformerSeparation

import argparse


#helper functions

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

#check similarity score of audio files
def verify_audio_model1(audio1, audio2):
    # Verify similarity between two audio files using model 1
    audio1_data = audio1[1] / np.max(np.abs(audio1[1]))
    audio2_data = audio2[1] / np.max(np.abs(audio2[1]))
    audio1_tensor = torch.tensor(audio1_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    audio2_tensor = torch.tensor(audio2_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    similarity = verification(model_1, audio1_tensor, audio2_tensor)
    return str(similarity.cpu().detach().numpy()[0])

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialise model
model = init_model("wavlm_base_plus", checkpoint="/DATA/arora8/SpeechUnderstanding/PA2/Q1/checkpoints/wavelmbase+/WavLM-Base+.pt").to(device)
model.eval()

# Gradio interface for audio verification using model 1
audio1_input = gr.components.Audio(source="upload", label="Upload test sample 1", type="numpy")
audio2_input = gr.components.Audio(source="upload", label="Upload test sample 2", type="numpy")
similarity_output = gr.outputs.Label(label="s=Simimlarity Score")
gr.Interface(fn=verify_audio_model1, inputs=[audio1_input, audio2_input], outputs=similarity_output).launch()

        