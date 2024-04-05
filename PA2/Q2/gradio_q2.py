#required imports
import gradio as gr
import numpy as np
import torch

from speechbrain.pretrained import SepformerSeparation as seperator

import argparse


#helper functions

#function for separating audio
def separate_audio_model2(audio):
    # Separate audio using the SepformerSeparation model
    sr = audio[0]
    audio_data = audio[1] / np.max(np.abs(audio[1]))
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to('cuda')
    extracted_audios = model_2.separate_batch(audio_tensor)
    audio1 = extracted_audios[:,:,0].cpu().detach().numpy()
    audio2 = extracted_audios[:,:,1].cpu().detach().numpy()
    return (sr, audio1[0]), (sr, audio2[0])

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_pretrainedmodel = "path to checkpoint"

# Load model 2 for audio separation
model_2 = seperator.from_hparams(source=path_to_pretrainedmodel,
    run_opts={"device": device}
).to(device)

# Gradio interface for audio separation using model 2
audio_input = gr.components.Audio(source="upload", label="Upload Mixed Audio (in .wav format)", type="numpy")
audio_outputs = [gr.outputs.Audio(label="Audio 1", type="numpy"), gr.outputs.Audio(label="Audio 2", type="numpy")]
gr.Interface(fn=separate_audio_model2, inputs=audio_input, outputs=audio_outputs).launch()