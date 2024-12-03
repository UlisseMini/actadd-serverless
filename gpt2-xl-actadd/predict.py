from cog import BasePredictor, Input
from transformer_lens import HookedTransformer
import torch
from functools import partial
from typing import List, Dict


CACHE_DIR = "weights"
MODEL_NAME = "gpt2-xl"


# Global model
model = None


tlen = lambda prompt: model.to_tokens(prompt).shape[1]
pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))

def pad_both(p_add, p_sub):
  l = max(tlen(p_add), tlen(p_sub))
  return pad_right(p_add, l), pad_right(p_sub, l)


def get_resid_pre(prompt: str, layer: int):
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


def get_act_diff(prompt_add: str, prompt_sub: str, layer: int):
    act_add = get_resid_pre(prompt_add, layer)
    act_sub = get_resid_pre(prompt_sub, layer)
    return act_add - act_sub # if this errors you forgot to pad


def ave_hook(act_diff, resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return  # caching in model.generate for new tokens

    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    resid_pre[:, :apos, :] += act_diff


def generate_hooked(prompt_batch: List[str], prompt_add: str, prompt_sub: str, layer: int, coeff: float, max_new_tokens: int, sampling_kwargs: Dict[str, float], verbose=False, seed=None):
    prompt_add, prompt_sub = pad_both(prompt_add, prompt_sub)
    act_diff = coeff*get_act_diff(prompt_add, prompt_sub, layer)
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", partial(ave_hook, act_diff))]

    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=editing_hooks):
        tokenized = model.to_tokens(prompt_batch)
        res_tokens: List[torch.Tensor] = model.generate(
            input=tokenized, do_sample=True, verbose=verbose, **sampling_kwargs, max_new_tokens=max_new_tokens,
        )
        res_strs: List[str] = model.to_string(res_tokens[:, 1:])

    return [x.replace("<|endoftext|>", "") for x in res_strs]


"""
import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-xl"
CACHE_DIR = "weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device=device, local_files_only=True)
"""


class Predictor(BasePredictor):
    def setup(self):
        global model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        model = HookedTransformer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device=device, local_files_only=True)
        model.eval()
        model.to(device)

    def predict(
        self,
        prompt_batch: List[str] = Input(description="The prompts to generate from"),
        coeff: float = Input(description="The coefficient for the activation addition", default=0.0),
        layer: int = Input(description="The layer to apply the actadd to"),
        prompt_add: str = Input(description="The positive part of the actadd"),
        prompt_sub: str = Input(description="The negative part of the actadd"),
        max_new_tokens: int = Input(description="The maximum number of new tokens to generate", default=50),
        seed: int = Input(description="Random seed for reproducibility", default=None),
        temperature: float = Input(description="Temperature for generation", default=1.0),
        top_p: float = Input(description="Top-p for generation", default=0.3),
        freq_penalty: float = Input(description="Frequency penalty for generation", default=1.0),
        # TODO: Add padding method
    ) -> List[str]:
        if model is None: # for typing, cog prob handles this
            raise ValueError("Model is None, setup not finished yet.")

        # TODO: Handle prompt_batch too large for GPU memory (copy from old notebook)
        sampling_kwargs = dict(temperature=temperature, top_p=top_p, freq_penalty=freq_penalty)
        generations: List[str] = generate_hooked(
            prompt_batch=prompt_batch,
            prompt_add=prompt_add,
            prompt_sub=prompt_sub,
            layer=layer,
            coeff=coeff,
            max_new_tokens=max_new_tokens,
            seed=seed,
            sampling_kwargs=sampling_kwargs,
        )
        return generations
