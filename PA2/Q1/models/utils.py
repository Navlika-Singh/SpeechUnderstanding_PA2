import torch
import fairseq
from packaging import version
import torch.nn.functional as F
from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from s3prl.upstream.interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    # state = load_checkpoint_to_cpu(filepath)
    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
