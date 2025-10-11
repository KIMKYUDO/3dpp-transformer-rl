import numpy as np
import torch

def get_rng_state():
    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def set_rng_state(state):
    if "torch" in state and state["torch"] is not None:
        t_state = state["torch"]
        if isinstance(t_state, (list, np.ndarray)):
            t_state = torch.ByteTensor(t_state)
        elif isinstance(t_state, torch.Tensor) and not isinstance(t_state, torch.ByteTensor):
            t_state = torch.ByteTensor(t_state.cpu().numpy())
        torch.set_rng_state(t_state)

    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])

    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = []
        for s in state["cuda"]:
            if isinstance(s, (list, np.ndarray)):
                s = torch.ByteTensor(s)
            elif isinstance(s, torch.Tensor) and not isinstance(s, torch.ByteTensor):
                s = torch.ByteTensor(s.cpu().numpy())
            cuda_states.append(s)
        torch.cuda.set_rng_state_all(cuda_states)
