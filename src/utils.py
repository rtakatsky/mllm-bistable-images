import glob
import os
import math
import json

# Project paths are derived from the location of this file so the repo can be
# moved between workspaces without editing any code. Override `VLM_BISTABLE_HF_CACHE_DIR`
# (or HF_HOME) in your environment if you want the HuggingFace cache to live
# outside the project tree.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
HF_CACHE_DIR = os.environ.get(
    "VLM_BISTABLE_HF_CACHE_DIR",
    os.path.join(PROJECT_ROOT, "huggingface_cache"),
)
# Must be set before transformers is imported (it reads HF_HOME at import time).
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor
    )
from typing import List, Dict, Optional, Union, Literal, Tuple, Any
import random
import numpy as np
from tqdm import tqdm
import argparse
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from collections import defaultdict

# preparing images
class RotatedRGBAImageDataset(Dataset):
    def __init__(self, img_path, angles, output_size=(336,336), bg_color=(255,255,255)):
        """Rotated views of an RGBA image, rendered on a canvas enlarged by sqrt(2)
        so that no rotation crops the image.

        Args:
            img_path (str): path to the RGBA image
            angles (list of int): rotation angles in degrees, e.g. [0, 30, 60, ...]
            output_size (tuple): (width, height) of the returned images
            bg_color (tuple): background colour (R, G, B)
        """
        self.base = Image.open(img_path).convert("RGBA")
        self.angles = angles
        self.output_size = output_size
        self.bg_color = bg_color
  # original image size
        w, h = self.base.size
  # Enlarge the canvas to the diagonal so that rotation never crops the image.
        diag = math.sqrt(w**2 + h**2)
        self.canvas_size = (math.ceil(diag), math.ceil(diag))

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        angle = self.angles[idx]
        canvas = Image.new("RGBA", self.canvas_size, (0,0,0,0))
        x = (self.canvas_size[0] - self.base.width)//2
        y = (self.canvas_size[1] - self.base.height)//2
        canvas.paste(self.base, (x, y), mask=self.base)
        rotated = canvas.rotate(angle, resample=Image.BICUBIC, expand=False)
        bg = Image.new("RGBA", self.canvas_size, self.bg_color+(255,))
        composite = Image.alpha_composite(bg, rotated).convert("RGB")

        if self.output_size is not None:
            composite = composite.resize(self.output_size, resample=Image.BICUBIC)
        return composite
    
# preparing VLMs

VLM_DICT={
    # llava models
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
    "llava-v1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf",

    # qwen models
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",

    # smolvlm models
    "smolvlm-2b": "HuggingFaceTB/SmolVLM-Instruct",

    # idefics models
    "idefics2-8b": "HuggingFaceM4/idefics2-8b",

    # blip models
    "instructblip-vicuna-7b": "Salesforce/instructblip-vicuna-7b",

}

DUCK_RABBIT_TOKENS = {
    "llava-1.5-7b": [["▁bird"], ["▁rabb"]],
    "llava-1.5-13b": [["▁bird"], ["▁rabb"]],
    "llava-v1.6-vicuna-7b": [["▁bird"], ["▁rabb"]],
    "llava-v1.6-mistral-7b": [["▁bird"], ["▁rab"]],
    "llama3-llava-next-8b": [["Ġbird"], ["Ġrabbit"]],
    "Qwen2-VL-2B-Instruct": [["Ġbird"], ["Ġrabbit"]],
    "Qwen2-VL-7B-Instruct": [["Ġbird"], ["Ġrabbit"]],
    "smolvlm-2b": [["Ġbird"], ["Ġrabbit"]],
    "idefics2-8b": [["▁bird"], ["▁rab"]],
    "instructblip-vicuna-7b": [["▁bird"], ["▁rabb"]],
}

# Raven--bear figure-ground image.
# LLaVA-family models tend to report "bird"/"dog"; non-LLaVA models tend to
# report "bird"/"bear". Token list: [<bird-side>, <other-side>].
RAVEN_BEAR_TOKENS = {
    "llava-1.5-7b": [["▁bird"], ["▁dog"]],
    "llava-1.5-13b": [["▁bird"], ["▁dog"]],
    "llava-v1.6-vicuna-7b": [["▁bird"], ["▁dog"]],
    "llava-v1.6-mistral-7b": [["▁bird"], ["▁dog"]],
    "llama3-llava-next-8b": [["Ġbird"], ["Ġdog"]],
    "Qwen2-VL-2B-Instruct": [["Ġbird"], ["Ġbear"]],
    "Qwen2-VL-7B-Instruct": [["Ġbird"], ["Ġbear"]],
    "smolvlm-2b": [["Ġbird"], ["Ġbear"]],
    "idefics2-8b": [["▁bird"], ["▁bear"]],
    "instructblip-vicuna-7b": [["▁bird"], ["▁bear"]],
}

def _get_token_pair(token_dict: Dict[str, Any], model_id: str, label: str) -> Tuple[str, str]:
    if model_id not in token_dict:
        raise ValueError(f"No {label} token mapping found for model: {model_id}")
    x_tokens, y_tokens = token_dict[model_id]
    if len(x_tokens) != 1 or len(y_tokens) != 1:
        raise ValueError(
            f"{label}: helper expects single tokens, got {x_tokens} vs {y_tokens} for {model_id}."
        )
    return x_tokens[0], y_tokens[0]


def get_raven_bear_tokens(model_id: str) -> Tuple[str, str]:
    return _get_token_pair(RAVEN_BEAR_TOKENS, model_id, "raven_bear")


def get_duck_rabbit_tokens(model_id: str) -> Tuple[str, str]:
    if model_id not in DUCK_RABBIT_TOKENS:
        raise ValueError(f"No duck/rabbit token mapping found for model: {model_id}")

    x_tokens, y_tokens = DUCK_RABBIT_TOKENS[model_id]

    if len(x_tokens) != 1 or len(y_tokens) != 1:
        raise ValueError(
            "Current decision_boundary_from_topk expects single tokens. "
            f"Got {x_tokens} vs {y_tokens} for {model_id}."
        )

    return x_tokens[0], y_tokens[0]

def collate_fn(batch):
    """
    batch: list of PIL.Image
    returns dict with images list
    """
    return {'pixel_values': batch}

def ensure_vocab_size_for_generation(model, processor=None):
    """
    Some multimodal HF configs, e.g. LlavaNextConfig, do not expose
    config.vocab_size at the top level, but GenerationMixin utilities
    such as compute_transition_scores expect it.
    """
    if hasattr(model.config, "vocab_size"):
        return

    candidates = []

    if hasattr(model.config, "text_config"):
        candidates.append(getattr(model.config.text_config, "vocab_size", None))

    if hasattr(model.config, "language_config"):
        candidates.append(getattr(model.config.language_config, "vocab_size", None))

    if hasattr(model, "language_model"):
        candidates.append(getattr(model.language_model.config, "vocab_size", None))

    if processor is not None and hasattr(processor, "tokenizer"):
        candidates.append(len(processor.tokenizer))

    for vocab_size in candidates:
        if vocab_size is not None:
            model.config.vocab_size = int(vocab_size)
            return

    raise AttributeError(
        "Could not infer vocab_size for this model. "
        "Please inspect model.config and model.language_model.config."
    )

def run_vlm_sampling(
    model: Union[LlavaForConditionalGeneration, LlavaNextForConditionalGeneration],
    processor: Union[LlavaProcessor, LlavaNextProcessor],
    dataset: Dataset,
    prompt: str,
    sample_n: int = 10,
    batch_size: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
    use_logit: bool = False,
    top_k: int = 100,
    prompt_format: str = None,
    prompt_prefix: str = None,
    num_beams: Optional[int] = None,
    extra_tokens: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Run a VLM on every image of `dataset` with the processor's chat template.

    Args:
        model: the loaded VLM
        processor: its AutoProcessor
        dataset: a Dataset returning PIL images
        prompt: the user text prompt
        batch_size: images per forward pass
        device: torch.device (auto-detected when None)

    Returns:
        one record per image (generated texts and/or first-token probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    
    if prompt_format is None:
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image'}
                ],
            },
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        text_prompt = prompt_format.format(prompt=prompt)

    if prompt_prefix is not None:
        text_prompt = text_prompt + prompt_prefix


    results: List[Dict[str, object]] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
  # Process batch by batch; images within a batch are generated in parallel.
        for batch_idx, batch in enumerate(loader):
            images = batch['pixel_values']
  # angles of this batch
            start = batch_idx * batch_size
            batch_angles = dataset.angles[start: start + len(images)]
  # one output list per image
            outputs_per_img = {angle: [] for angle in batch_angles}

  # sample the whole batch at once
            texts = [text_prompt] * len(images)
            inputs = processor(
                images=images,
                text=texts,
                return_tensors='pt'
            ).to(device)

            if use_logit:
                tok = processor.tokenizer
                extra_tokens = extra_tokens or []
                vocab = tok.get_vocab()
                unk_id = getattr(tok, "unk_token_id", None)
                unk_tok = getattr(tok, "unk_token", None)

                extra_tok_ids: List[int] = []
                extra_tok_strs: List[str] = []
                for t in extra_tokens:
                    tid = vocab.get(t, None)
                    if tid is None:
                        # fallback: convert_tokens_to_ids may return unk_id for missing
                        cand = tok.convert_tokens_to_ids(t)
                        if isinstance(cand, int) and cand != unk_id:
                            tid = cand
                        elif unk_id is not None and cand == unk_id and t == unk_tok:
                            tid = cand  # allow explicit <unk>
                        else:
                            tid = None
                    if tid is not None:
                        extra_tok_ids.append(int(tid))
                        extra_tok_strs.append(t)

                gen_out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
                first_token_logits = gen_out.scores[0]  # (B, V)
                probs_all = torch.softmax(first_token_logits, dim=-1)  # (B, V)

                top_values, top_indices = first_token_logits.topk(k=top_k, dim=-1)  # (B, K)
                top_probs = probs_all.gather(dim=-1, index=top_indices)              # (B, K)

                for row_i, angle in enumerate(batch_angles):
                    # top-k base
                    idx_list = top_indices[row_i].detach().cpu().tolist()
                    tok_list = tok.convert_ids_to_tokens(idx_list)
                    logit_list = top_values[row_i].detach().cpu().tolist()
                    prob_list  = top_probs[row_i].detach().cpu().tolist()

                    # --- NEW: append extra tokens if missing ---
                    present_ids = set(idx_list)
                    for t_str, t_id in zip(extra_tok_strs, extra_tok_ids):
                        if t_id in present_ids:
                            continue
                        present_ids.add(t_id)
                        tok_list.append(t_str)  # store exactly what user passed
                        idx_list.append(t_id)
                        logit_list.append(float(first_token_logits[row_i, t_id].detach().cpu()))
                        prob_list.append(float(probs_all[row_i, t_id].detach().cpu()))

                    outputs_per_img[angle].append({
                        "tokens": tok_list,
                        "indices": idx_list,
                        "logits":  logit_list,
                        "probs":   prob_list,
                    })
            else:
                if num_beams is not None:

                    ensure_vocab_size_for_generation(model, processor)  
                    # how many sequences to return per image
                    num_return = min(top_k, num_beams)  # HF requires num_return_sequences <= num_beams

                    gen_out = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=num_beams,
                        num_return_sequences=num_return,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        length_penalty=0.0,
                        early_stopping=False,
                    )

                    prompt_len = inputs.input_ids.shape[1]
                    gen_token_ids = gen_out.sequences[:, prompt_len:]  # (B*num_return, <=max_new_tokens)
                    decoded = processor.batch_decode(gen_token_ids, skip_special_tokens=True)

                    # Per-token logprobs for the chosen tokens (aligned to final sequences)
                    beam_indices = getattr(gen_out, "beam_indices", None)
                    token_logprobs = model.compute_transition_scores(
                        gen_out.sequences,
                        gen_out.scores,
                        beam_indices=beam_indices,
                        normalize_logits=True,   # log-softmax
                    )  # (B*num_return, gen_len)

                    # Continuation probability = probability of the TEXT tokens only:
                    # exclude the EOS emission term and everything after it. This is a
                    # uniform convention for finished and unfinished beams and matches
                    # HF's own beam ranking (sequences_scores). Without it, IDEFICS2 —
                    # whose tokenizer EOS (</s>) is an unnatural stop for the model
                    # (logp ~ -18; its native stop is <end_of_utterance>) — had every
                    # finished beam scaled by ~1e-8.
                    # Stored "indices" are truncated to the text tokens accordingly, so
                    # teacher-forced rescoring of pooled candidates is EOS-free too.
                    eos_ids = set()
                    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                    if isinstance(gc_eos, (list, tuple)):
                        eos_ids.update(int(x) for x in gc_eos)
                    elif gc_eos is not None:
                        eos_ids.add(int(gc_eos))
                    for tid in (processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id):
                        if tid is not None:
                            eos_ids.add(int(tid))
                    ids_all = gen_token_ids.detach().cpu().tolist()
                    lp_all = token_logprobs.detach().cpu()
                    seq_logprob_list, ids_list = [], []
                    for row, ids in enumerate(ids_all):
                        cut = next((k for k, t in enumerate(ids) if t in eos_ids), len(ids))
                        keep = max(cut, 1)  # an immediately-ending beam keeps its EOS term
                        seq_logprob_list.append(float(lp_all[row, :keep].sum()))
                        ids_list.append(ids[:cut])

                    # group results back per image, in "use_logit-like" format
                    for i, angle in enumerate(batch_angles):
                        s = slice(i * num_return, (i + 1) * num_return)
                        outputs_per_img[angle].append({
                            "tokens": decoded[s],  # list[str] (each is a whole sequence)
                            "indices": ids_list[s],  # list[list[int]] (text tokens only, EOS excluded)
                            "logits": seq_logprob_list[s],    # list[float] (log prob of the text tokens)
                            "probs": [float(np.exp(x)) for x in seq_logprob_list[s]],  # list[float]
                            # optional if you ever want token-level:
                            # "token_logprobs": token_logprobs[s].detach().cpu().tolist(),
                        })

                    del gen_out, gen_token_ids
                    torch.cuda.empty_cache()
                else:
                    for n in range(sample_n):
                        torch.manual_seed(int(seed + n))

                        gen_ids = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=temperature,
                            # Explicit HF defaults: Qwen2-VL ships top_k=1 /
                            # top_p=0.001 in its generation_config.json, which
                            # silently collapses sampling to greedy decoding
                            # unless overridden here. No-op for other models.
                            top_k=50,
                            top_p=1.0,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=len(images),
                            use_cache=True,
                            pad_token_id=processor.tokenizer.eos_token_id,
                        )
                        gen_ids = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
                        ]
                        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

  # distribute to the images of the batch
                        for angle, text_out in zip(batch_angles, decoded):
                            outputs_per_img[angle].append(text_out)

                        del gen_ids
                        torch.cuda.empty_cache()

  # collect the results
            for angle in batch_angles:
                results.append({'angle': angle, 'output': outputs_per_img[angle]})
            
            del inputs
            torch.cuda.empty_cache()
    
    return results

def slugify(text: str) -> str:
  # Lower-case and keep only alphanumerics and hyphens.
    import re
    s = text.lower()
  # non-alphanumerics -> spaces
    s = re.sub(r'[^a-z0-9]+', ' ', s)
  # runs of spaces -> hyphens
    s = re.sub(r'\s+', '-', s).strip('-')
    return s

def _patch_qwen2vl_beam_expansion(cls):
    """Backport of the upstream fix for beam search with Qwen2-VL under
    transformers==4.48 (HF PR #36083 / issue #39723).

    Qwen2-VL's pixel_values are a batch-less flattened patch sequence
    (n_patches, patch_dim), but GenerationMixin._expand_inputs_for_generation
    repeat_interleaves EVERY tensor kwarg along dim 0 as if it were a batch
    dim. For num_beams>1 that interleaves PATCHES, so each expanded "image"
    the vision tower reconstructs (chunked by the repeated grid_thw) is one
    quarter-strip of the original with every patch smeared 4x — beam 0, whose
    scores drive step 0 and whose KV cache all surviving beams inherit, sees
    only the (often empty) top strip. Validated fix: tile the whole visual
    block expand_size times instead (correct for batch_size==1, which is all
    our experiments use; a guard below refuses larger batches).
    """
    if getattr(cls, "_beam_expansion_patched", False):
        return
    orig = cls._expand_inputs_for_generation

    def fixed(expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        visual = {}
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if model_kwargs.get(k) is not None:
                visual[k] = model_kwargs.pop(k)
        if visual and expand_size > 1 and input_ids is not None and input_ids.shape[0] != 1:
            raise RuntimeError(
                "Qwen2-VL beam-expansion patch only supports batch_size==1 "
                f"(got batch {input_ids.shape[0]}); run with batch_size=1."
            )
        input_ids, model_kwargs = orig(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids, **model_kwargs,
        )
        for k in ("pixel_values", "pixel_values_videos"):
            if k in visual:
                model_kwargs[k] = torch.cat([visual[k]] * expand_size, dim=0)
        for k in ("image_grid_thw", "video_grid_thw"):
            if k in visual:
                model_kwargs[k] = visual[k].repeat(expand_size, 1)
        return input_ids, model_kwargs

    cls._expand_inputs_for_generation = staticmethod(fixed)
    cls._beam_expansion_patched = True
    print("[PATCH] Qwen2-VL beam-search visual-input expansion fix applied (transformers 4.48)")


def load_vlm(model_id, trust_remote_code: bool = True, torch_dtype=torch.float16, device_map=None, use_fast=True, return_prompt_format: bool = False):
    from transformers import AutoProcessor

    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    prompt_format = None
    if "llava-v1.6" in model_id or "llava-next" in model_id:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained(
            model_id,
            use_fast=use_fast,
            **load_kwargs
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs
        )
    elif "llava" in model_id:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        processor = LlavaProcessor.from_pretrained(
            model_id,
            use_fast=use_fast,
            **load_kwargs
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs
        )
    # —— Qwen2‑VL‑7B‑Instruct ——
    elif "qwen2-vl" in model_id.lower():
        from transformers import Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_id, use_fast=use_fast, **load_kwargs
        )  # Qwen2‑VL vision preprocessing :contentReference[oaicite:3]{index=3}
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )  # Qwen2‑VL generation head :contentReference[oaicite:4]{index=4}
        _patch_qwen2vl_beam_expansion(Qwen2VLForConditionalGeneration)
    # —— SmolVLM‑2B‑Instruct ——  
    elif "SmolVLM" in model_id or "smolvlm" in model_id:
        from transformers import AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    elif "idefics2" in model_id.lower():

        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

        model = model_cls.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    
    elif "instructblip-vicuna-7b" in model_id.lower():
        from transformers import (
            InstructBlipProcessor,
            InstructBlipForConditionalGeneration,
        )

        processor = InstructBlipProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # InstructBLIP is usually not chat-template style.
        # It takes text directly as an instruction.
        prompt_format = "{prompt}"

    # —— IDEFICS‑9B‑Instruct ——  
    elif "idefics-9b-instruct" in model_id:
        from transformers import IdeficsForVisionText2Text

        processor = AutoProcessor.from_pretrained(
            model_id, **load_kwargs
        )  # IDEFICS processor for interleaved image+text :contentReference[oaicite:5]{index=5}
        model = IdeficsForVisionText2Text.from_pretrained(
            model_id, **load_kwargs
        )  # IDEFICS vision‑text2text model :contentReference[oaicite:6]{index=6}

    else:
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=use_fast,
            **load_kwargs
        )
        config = AutoConfig.from_pretrained(
            model_id,
            **load_kwargs
        )
        if getattr(config, "is_encoder_decoder", False):
            # most captioning / encoder-decoder VLMs
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs)
        else:
            # chatty / causal LMs
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if not return_prompt_format:
        return processor, model
    else:
        if "llava-v1.6-mistral-7b" in model_id:
            prompt_format="[INST] <image>\n{prompt} [/INST]"
        elif "llava-v1.6-vicuna-7b" in  model_id or "llava-v1.6-vicuna-13b" in model_id:
            prompt_format="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
        elif "llava-v1.6-34b" in model_id:
            prompt_format="<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
        elif "llama3-llava-next-8b" in model_id:
            prompt_format="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "llava-1.5-7b" in model_id or "llava-1.5-13b" in model_id:
            prompt_format="USER: <image>\n{prompt} ASSISTANT:"
        
        return processor, model, prompt_format

# general description analysis
COLOR_DICT = {
        "▁du": "blue",
        "▁Duck": "blue",
        "▁duck": "blue", # mistral
        "Ġduck": "blue", # llama
        "▁bird": "skyblue",
        "Ġbird": "skyblue", # llama
        "▁rabb": "orange",
        "▁Rab": "orange",
        "▁rab": "orange", # mistral
        "Ġrabbit": "orange", # llama
        "▁b": "salmon",
        "▁Bun": "salmon",
        "Ġbunny": "salmon", # llama
        "▁drawing": "gray",
        "Ġdrawing": "gray", # llama
        "▁fish": "green",
        "Ġfish": "green", # llama
    }

# single-token description analysis
import json
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.image as mpimg
def extract_topk_from_file(
    input_path: str,
    k: int,
    extra_tokens: List[str] = [],
    anchor_angle: Optional[float] = 0,   # NEW: choose top-k from this angle
) -> List[Dict[str, Any]]:
    """
    Read JSON and:
      - choose top-k tokens from anchor_angle output (or first valid output if not found)
      - collect those token probabilities for all angles in the file
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1) choose anchor output for top-k token selection
    anchor_output = None
    if anchor_angle is not None:
        for item in data:
            if item.get("angle") == anchor_angle and item.get("output"):
                anchor_output = item["output"][0]
                break

    # fallback: first valid output
    if anchor_output is None:
        for item in data:
            if item.get("output"):
                anchor_output = item["output"][0]
                break

    if not anchor_output:
        raise ValueError(f"No valid output found in JSON: {input_path}")

    # 1-a) sort by prob and take top-k
    zpairs = sorted(
        zip(anchor_output.get('tokens', []), anchor_output.get('probs', [])),
        key=lambda tp: tp[1],
        reverse=True
    )[:k]
    topk_tokens = [tok for tok, _ in zpairs]

    # 1-b) append extra tokens if missing
    for tok in extra_tokens:
        if tok not in topk_tokens:
            topk_tokens.append(tok)

    # 2) collect probs for all angles
    topk_data = []
    for item in data:
        angle = item.get('angle')
        outputs = item.get('output', [])
        if not outputs:
            continue

        out = outputs[0]
        prob_map = dict(zip(out.get('tokens', []), out.get('probs', [])))
        probs = [prob_map.get(tok, np.nan) for tok in topk_tokens]

        topk_data.append({
            'angle': angle,
            'tokens': topk_tokens,
            'probs': probs,
        })

    return topk_data

import hashlib
import re

BEAM_QUALIFIERS = ("not", "specifically", "possibly", "maybe")

def classify_beam_continuation(s: str) -> str:
    """'enum' if the continuation enumerates a second interpretation, else 'excl'.

    Rule: qualifier words mark refinements/negations -> excl;
    'and'/'or' as a word, or a comma followed by content -> enum; everything
    else (endings, descriptors, possessives, empty) -> excl.
    """
    t = s.strip()
    if not t:
        return "excl"
    if re.search(r"\b(%s)\b" % "|".join(BEAM_QUALIFIERS), t):
        return "excl"
    if re.search(r"\b(and|or)\b", t):
        return "enum"
    if re.search(r",\s*\S", t):
        return "enum"
    return "excl"

# exclusive = greens/teals; enumeration = purples/magentas. Blues/oranges are
# reserved for the duck/bird vs rabbit token colors of the rotation plots
# (COLOR_DICT); reds are avoided as too close to salmon (bunny).
EXCL_PALETTE = ["#1b7f3b", "#2aa198", "#6b8e23", "#0e6f5c", "#52b788", "#00695c", "#98c93c", "#33691e"]
ENUM_PALETTE = ["#8e44ad", "#c71585", "#6a51a3", "#d33682", "#9c27b0", "#553c8b", "#ba68c8", "#7b1fa2"]
NEUTRAL_PALETTE = ["#555555", "#8c564b", "#2ca02c", "#9467bd", "#7f7f7f", "#6b8e23", "#553c8b", "#a0522d"]
CLASS_SUM_COLORS = {"exclusive": "#1b7f3b", "enumerating": "#8e44ad"}

def _stable_color(label: str, palette: List[str], used: set) -> str:
    """Deterministic label->color: the same label hashes to the same palette
    entry in every plot; a within-plot collision bumps to the next free entry."""
    idx = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % len(palette)
    for off in range(len(palette)):
        c = palette[(idx + off) % len(palette)]
        if c not in used:
            used.add(c)
            return c
    return palette[idx]


def plot_topk_tokens(
    topk_data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    image: Image.Image = None,
    color_dict: Optional[Dict[str, str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    center_angle: Optional[Union[int, float, List[Union[int, float]]]] = 0,
    beam_mode: bool = False,
    class_sum: bool = False,
    max_curves: Optional[int] = None,
) -> None:
    """
    Plot probability curves for top-k tokens over angles.

    Supports:
      1) single-run mode:
         topk_data = List[Dict]
         center_angle = int/float
      2) aggregate mode:
         topk_data = List[List[Dict]]   # multiple prob_data
         center_angle = List[int/float] # one center per run

    Aggregate mode plots:
      - each individual run as dashed line
      - mean curve as solid line
      - std as shaded area
    """
    if not topk_data:
        return

    # -----------------------------
    # Normalize inputs to "runs"
    # -----------------------------
    is_aggregate = isinstance(topk_data[0], list)  # List[List[Dict]] vs List[Dict]

    if is_aggregate:
        runs: List[List[Dict[str, Any]]] = topk_data  # type: ignore
        if isinstance(center_angle, (int, float)) or center_angle is None:
            centers = [0 if center_angle is None else float(center_angle)] * len(runs)
        else:
            centers = [0.0 if c is None else float(c) for c in center_angle]
            if len(centers) != len(runs):
                raise ValueError(
                    f"Length mismatch: got {len(runs)} runs but {len(centers)} center angles"
                )
    else:
        runs = [topk_data]  # type: ignore
        c = 0.0 if center_angle is None else float(center_angle if not isinstance(center_angle, list) else center_angle[0])
        centers = [c]

    # -----------------------------
    # Convert each run into:
    #   token -> {shifted_angle: prob}
    # -----------------------------
    per_run_series: List[Dict[str, Dict[float, float]]] = []
    all_angles = set()
    token_order = []   # preserve first-seen order
    seen_tokens = set()

    for run, c in zip(runs, centers):
        series = defaultdict(dict)
        sorted_data = sorted(run, key=lambda x: x["angle"])

        for item in sorted_data:
            ang = float(item["angle"]) - c
            all_angles.add(ang)

            toks = item.get("tokens", [])
            probs = item.get("probs", [])
            for tok, p in zip(toks, probs):
                series[tok][ang] = p
                if tok not in seen_tokens:
                    seen_tokens.add(tok)
                    token_order.append(tok)

        per_run_series.append(dict(series))

    if not all_angles:
        return

    angles = sorted(all_angles)

    # Optional token ordering tweak: prioritize color_dict order
    if color_dict:
        ordered = [tok for tok in color_dict.keys() if tok in seen_tokens]
        ordered += [tok for tok in token_order if tok not in set(ordered)]
        token_order = ordered

    # -----------------------------
    # Optional: collapse each run's curves into per-class probability sums
    # (exclusive vs enumerating continuations) -- used for beam aggregates.
    # -----------------------------
    if class_sum:
        collapsed = []
        for run_series in per_run_series:
            run_angles = set()
            for amap in run_series.values():
                run_angles.update(amap.keys())
            # a class with no tracked member is an explicit zero, not a gap:
            # tracked candidates are the top-probability continuations, so
            # anything untracked lies below them.
            sums: Dict[str, Dict[float, float]] = {k: {a: 0.0 for a in run_angles}
                                                   for k in CLASS_SUM_COLORS}
            for tok, amap in run_series.items():
                key = ("enumerating" if classify_beam_continuation(tok) == "enum"
                       else "exclusive")
                for a, p in amap.items():
                    if p == p:  # skip NaN
                        sums[key][a] = sums[key].get(a, 0.0) + p
            collapsed.append(sums)
        per_run_series = collapsed
        token_order = list(CLASS_SUM_COLORS.keys())

    # -----------------------------
    # Per-curve statistics (mean/std across runs)
    # -----------------------------
    stats = {}
    for tok in token_order:
        Y = np.full((len(per_run_series), len(angles)), np.nan, dtype=float)
        for i, run_series in enumerate(per_run_series):
            amap = run_series.get(tok, {})
            for j, a in enumerate(angles):
                if a in amap:
                    Y[i, j] = amap[a]
        if np.all(np.isnan(Y)):
            continue
        counts = np.sum(np.isfinite(Y), axis=0)
        sums = np.nansum(Y, axis=0)
        mean = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        var = np.divide(np.sum(diffs**2, axis=0), counts,
                        out=np.full_like(mean, np.nan, dtype=float), where=counts > 0)
        stats[tok] = (counts, mean, np.sqrt(var))

    token_order = [t for t in token_order if t in stats]

    # -----------------------------
    # Cap the number of curves: keep pinned (color_dict) tokens, rank the rest
    # by peak mean probability, drop the tail entirely.
    # -----------------------------
    if max_curves is not None and len(token_order) > max_curves:
        pinned = [t for t in token_order if color_dict and t in color_dict]
        rest = [t for t in token_order if t not in set(pinned)]
        rest.sort(key=lambda t: -float(np.nanmax(stats[t][1])))
        keep = set(pinned) | set(rest[:max(0, max_curves - len(pinned))])
        token_order = [t for t in token_order if t in keep]

    # -----------------------------
    # Resolve ONE color per curve up front; the same color is used for the
    # mean line and the std shade, and the same label hashes to the same
    # color in every plot (C3).
    # -----------------------------
    used_colors: set = set()
    tok_colors: Dict[str, str] = {}
    for tok in token_order:
        if class_sum:
            tok_colors[tok] = CLASS_SUM_COLORS[tok]
        elif color_dict and tok in color_dict:
            tok_colors[tok] = color_dict[tok]
            used_colors.add(color_dict[tok])
        elif beam_mode:
            pal = ENUM_PALETTE if classify_beam_continuation(tok) == "enum" else EXCL_PALETTE
            tok_colors[tok] = _stable_color(tok, pal, used_colors)
        else:
            tok_colors[tok] = _stable_color(tok, NEUTRAL_PALETTE, used_colors)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))

    for tok in token_order:
        counts, mean, std = stats[tok]
        color = tok_colors[tok]

        if len(per_run_series) > 1:
            valid_std = counts >= 2
            if np.any(valid_std):
                lower = np.clip(mean - std, 0, 1)
                upper = np.clip(mean + std, 0, 1)
                ax.fill_between(angles, lower, upper, where=valid_std,
                                alpha=0.18, color=color, linewidth=0)

        ax.plot(angles, mean, linewidth=2.0,
                marker="o" if len(angles) == 1 else None,
                label=tok, color=color)

    ax.set_xlabel("Angle")
    ax.set_ylabel("Probability")
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title, fontsize=15)
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    # Inset image (single image only; for aggregate mode just pass image=None)
    if image is not None:
        image_bbox = [0.39, 0.95, 0.2, 0.2]
        inset_ax = fig.add_axes(image_bbox, anchor="NE")
        inset_ax.imshow(image)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close()


def build_token_angle_map(
    topk_data: List[Dict[str, Any]],
    center_angle: float,
) -> Dict[float, Dict[str, float]]:
    """
    Convert one prob_data into:
      rel_angle -> {token: prob}
    where rel_angle = angle - center_angle
    """
    out: Dict[float, Dict[str, float]] = {}
    for item in topk_data:
        ang = float(item["angle"]) - float(center_angle)
        toks = item.get("tokens", [])
        probs = item.get("probs", [])
        out[ang] = dict(zip(toks, probs))
    return out


def plot_cue_effects_single(
    cue_to_record: Dict[str, Dict[str, Any]],
    cue_type: str,
    cue_order: List[str],
    color_dict: Dict[str, str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    One figure for one image (+ one base prompt + one cue_type).
    Mean/std is taken across relative angles.

    Visual encoding:
      - color: token
      - dashed thin lines: each relative-angle curve
      - solid line + shaded area: mean ± std across angles

    Canonical copy of the helper originally defined in
    top_down_cues.py, so other experiments (e.g. the
    figure-ground scripts) can render top-down cue plots in the same style.
    """
    if not cue_to_record:
        return

    x_labels = ["no cue"] + list(cue_order)
    x = np.arange(len(x_labels))
    token_order = list(color_dict.keys())

    # cue_maps[cue_label][rel_angle][token] = prob
    cue_maps: Dict[str, Dict[float, Dict[str, float]]] = {}
    all_rel_angles = set()

    for cue_label, rec in cue_to_record.items():
        prob_data = rec["prob_data"]
        center_angle = rec["center_angle"]
        amap = build_token_angle_map(prob_data, center_angle)
        cue_maps[cue_label] = amap
        all_rel_angles.update(amap.keys())

    if not all_rel_angles:
        return

    rel_angles = sorted(all_rel_angles)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for tok in token_order:
        # Y shape = [n_angles, n_cues]
        Y = np.full((len(rel_angles), len(x_labels)), np.nan, dtype=float)

        for i, rel_ang in enumerate(rel_angles):
            for j, cue in enumerate(x_labels):
                p = cue_maps.get(cue, {}).get(rel_ang, {}).get(tok, np.nan)
                Y[i, j] = p

        if np.all(np.isnan(Y)):
            continue

        color = color_dict.get(tok, None)

        # Dashed individual angle curves
        for i in range(Y.shape[0]):
            yi = Y[i]
            if np.all(np.isnan(yi)):
                continue
            ax.plot(
                x, yi,
                linestyle="--",
                linewidth=0.9,
                alpha=0.28,
                color=color,
                label="_nolegend_",
            )

        # Mean/std across angles
        counts = np.sum(np.isfinite(Y), axis=0)
        sums = np.nansum(Y, axis=0)
        mean = np.divide(
            sums, counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )

        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        var = np.divide(
            np.sum(diffs ** 2, axis=0),
            counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )
        std = np.sqrt(var)

        valid_std = counts >= 2
        if np.any(valid_std):
            lower = np.clip(mean - std, 0, 1)
            upper = np.clip(mean + std, 0, 1)
            ax.fill_between(
                x, lower, upper,
                where=valid_std,
                alpha=0.15,
                color=color,
                linewidth=0,
            )

        ax.plot(
            x, mean,
            marker="o",
            linewidth=2.0,
            color=color,
            label=tok,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{cue_type} cue")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq_h, uniq_l = [], []
    seen = set()
    for h, l in zip(handles, labels):
        if l == "_nolegend_" or l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax.legend(
            uniq_h, uniq_l,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


def decision_boundary_from_topk(
    topk_data: List[Dict[str, Any]],
    token_a: str,
    token_b: str,
    angle_lim: Tuple[float, float] = (-45, 45),
    eps: float = 1e-6,
    min_points: int = 5,
    clamp_to_range: bool = True,
    plot: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    # reject "almost stable" cases
    min_pred_range: float = 0.15,         # predicted max(p)-min(p) within angle_lim
    min_max_slope: float = 0.002,         # max |dp/dθ| (per degree); for logistic it's |a|/4
    # fitting knobs
    slope_grid: Optional[np.ndarray] = None,  # if None, uses a decent default
    denom_quantile: float = 0.0,          # optionally ignore lowest-denom points (0.0 keeps all)
    min_count: int = 3,
    thresholds: Tuple[float, float] = (0.4, 0.6),
) -> Optional[int]:
    """
    Fit a binary psychometric curve between token_a vs token_b using topk_data and
    return the decision boundary angle (integer). Uses a logistic fit in probability
    space via grid search (robust for saturated data).

    Returns None if:
      - not enough valid points
      - curve doesn't have enough range/slope (almost stable)
      - fitting degenerates
    """
    if not topk_data:
        return None

    lo, hi = angle_lim
    if lo > hi:
        lo, hi = hi, lo

    low_thr, high_thr = thresholds

    # Collect (angle, pA, pB)
    xs, pAs, pBs = [], [], []
    for item in topk_data:
        angle = item.get("angle", None)
        toks = item.get("tokens", [])
        probs = item.get("probs", [])
        if angle is None or not toks or not probs:
            continue
        angle = float(angle)
        if not (lo <= angle <= hi):
            continue

        pmap = dict(zip(toks, probs))
        pA = pmap.get(token_a, np.nan)
        pB = pmap.get(token_b, np.nan)

        xs.append(angle)
        pAs.append(pA)
        pBs.append(pB)

    x = np.asarray(xs, dtype=float)
    pA = np.asarray(pAs, dtype=float)
    pB = np.asarray(pBs, dtype=float)

    denom = pA + pB
    mask = np.isfinite(x) & np.isfinite(pA) & np.isfinite(pB) & np.isfinite(denom) & (denom > 0)
    if mask.sum() < min_points:
        return None

    x = x[mask]
    pA = pA[mask]
    pB = pB[mask]
    denom = denom[mask]

    # Optional: drop very low-evidence points where both tokens are tiny
    if denom_quantile > 0.0:
        thr = np.quantile(denom, denom_quantile)
        keep = denom >= thr
        if keep.sum() < min_points:
            return None
        x, pA, pB, denom = x[keep], pA[keep], pB[keep], denom[keep]

    # Binary-normalized probability for A
    p = pA / denom
    p = np.clip(p, eps, 1.0 - eps)

    # --- FIT: logistic in probability space via grid search (minimal but robust) ---
    # p_fit(angle) = sigmoid(a*(angle - mu))
    if slope_grid is None:
        # slopes (per degree): small -> gradual, large -> step-like
        slope_grid = np.concatenate([
            np.linspace(0.02, 0.2, 20),
            np.linspace(0.25, 2.5, 30),
        ])

    # Candidate boundaries: integer angles in-window
    mu_grid = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1, dtype=float)

    # Weights: denom (more weight when either token is relatively likely)
    w = denom.astype(float)
    w = w / (w.sum() + 1e-12)

    best_loss = np.inf
    best_mu = None
    best_a = None

    # vectorized-ish loop: iterate mu, compute best slope
    for mu in mu_grid:
        dx = x - mu  # (N,)
        # For each slope, compute p_pred
        # shape: (S, N)
        z = slope_grid[:, None] * dx[None, :]
        p_pred = 1.0 / (1.0 + np.exp(-z))
        # weighted MSE in probability space
        loss = np.sum(w[None, :] * (p_pred - p[None, :])**2, axis=1)  # (S,)
        j = int(np.argmin(loss))
        if loss[j] < best_loss:
            best_loss = float(loss[j])
            best_loss = best_loss
            best_mu = float(mu)
            best_a = float(slope_grid[j])

    if best_mu is None or best_a is None or not np.isfinite(best_mu) or not np.isfinite(best_a):
        return None

    # Reject "almost stable everywhere" using predicted range and max slope
    grid = np.linspace(lo, hi, 400)
    p_fit = 1.0 / (1.0 + np.exp(-(best_a * (grid - best_mu))))
    pred_range = float(np.max(p_fit) - np.min(p_fit))
    max_slope = float(abs(best_a) / 4.0)  # logistic max derivative

    if pred_range < min_pred_range or max_slope < min_max_slope:
        if plot:
            _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)
        return None
    
    n_low  = int(np.sum(p <= low_thr))
    n_high = int(np.sum(p >= high_thr))
    if n_low < min_count or n_high < min_count:
        # optionally still plot to debug
        if plot:
            _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)
        return None


    boundary = int(round(best_mu))
    if clamp_to_range:
        boundary = int(np.clip(boundary, int(np.min(x)), int(np.max(x))))

    if plot:
        _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)

    return boundary


def _plot_fit_debug(
    x: np.ndarray,
    p: np.ndarray,
    denom: np.ndarray,
    grid: np.ndarray,
    p_fit: np.ndarray,
    mu: float,
    token_a: str,
    token_b: str,
    lo: float,
    hi: float,
    title: Optional[str],
    save_path: Optional[str],
):
    order = np.argsort(x)
    xs = x[order]
    ps = p[order]
    ws = denom[order]

    fig, ax = plt.subplots(figsize=(8, 5))

    # marker size scaled by denom (optional but very informative)
    s = 20 + 180 * (ws / (np.max(ws) + 1e-12))
    ax.scatter(xs, ps, s=s, label="data (pA/(pA+pB))")

    ax.plot(grid, p_fit, label="fitted logistic (prob-space)")

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axvline(mu, linestyle="--", linewidth=1, label=f"steepest/boundary ≈ {mu:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Angle")
    ax.set_ylabel(f"P({token_a}) / (P({token_a})+P({token_b}))")
    ax.set_title(title or f"Psychometric fit: {token_a} vs {token_b}")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


def get_boundary(
    boundaries: Dict[str, Any],
    model_name: str,
    image_name: str,
    prompt_slug: str,
) -> Optional[int]:
    """
    Read boundary for a specific (model, image_name, prompt_slug).

    Returns:
      int boundary if found
      None if not found OR explicitly null
    """
    # Expected new schema: boundaries[model][image][prompt_slug]["boundary"]
    m = boundaries.get(model_name, None)
    if not isinstance(m, dict):
        print(f"[DBG] model not found in boundaries: {model_name}")
        return None

    bimg = m.get(image_name, None)
    if not isinstance(bimg, dict):
        print(f"[DBG] image not found under model={model_name}: {image_name}")
        return None

    entry = bimg.get(prompt_slug, None)
    if not isinstance(entry, dict):
        print(f"[DBG] prompt_slug not found under model={model_name}, image={image_name}: {prompt_slug}")
        print(f"[DBG] available prompt_slugs: {list(bimg.keys())[:5]}{'...' if len(bimg) > 5 else ''}")
        return None

    boundary = entry.get("boundary", None)
    if boundary is None:
        print(f"[DBG] boundary exists but is null for model={model_name}, image={image_name}, prompt_slug={prompt_slug}")
        return None

    return int(boundary)


def centered_va_image(
    raw_image_path: str,
    boundaries: Dict[str, Any],
    model_name: str,
    image_name: str,
    prompt_slug: str,
    output_size: Tuple[int, int] = (336, 336),
) -> Optional[Image.Image]:
    """Return the boundary-centered PIL image for a VA image, in memory.

    Looks up `boundaries[model_name][image_name][prompt_slug]["boundary"]` and
    returns `RotatedRGBAImageDataset(raw_image_path, [boundary])[0]`. Use this
    instead of reading a pre-centered file from `data/duck_rabbit_preprocessed/`,
    so the centered image is recomputed deterministically from the raw source
    and never written back to disk.

    Returns None if the boundary entry is missing or null.
    """
    boundary = get_boundary(boundaries, model_name, image_name, prompt_slug)
    if boundary is None:
        return None
    dataset = RotatedRGBAImageDataset(raw_image_path, [int(boundary)], output_size=output_size)
    return dataset[0]


import math
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class RedCircleImageDataset(Dataset):
    def __init__(
        self,
        img_path,
        radius_ratio=1/14,
        stride_ratio=1/14,
        line_width=4,
        bg_color=(255, 255, 255),
        include_baseline=True,
        output_size=(336,336),
    ):
        # Accept either a path or a PIL image so callers can pass in-memory
        # centered images (see `centered_va_image`) without a temp file.
        if isinstance(img_path, Image.Image):
            self.base = img_path.convert("RGBA")
        else:
            self.base = Image.open(img_path).convert("RGBA")
        self.radius_ratio = radius_ratio
        self.stride_ratio = stride_ratio
        self.line_width = line_width
        self.bg_color = bg_color
        self.include_baseline = include_baseline
        self.output_size = output_size
        w, h = self.base.size
        assert w == h, "Image must be square"
        self.image_size = w

        self.circle_radius = max(self.image_size * self.radius_ratio, 1.0)
        self.stride = max(self.image_size * self.stride_ratio, 1.0)

        xs = []
        x = self.stride
        while x < self.image_size - 1:
            xs.append(x)
            x += self.stride

        ys = []
        y = self.stride
        while y < self.image_size - 1:
            ys.append(y)
            y += self.stride

        self.centers = [(int(round(x)), int(round(y))) for y in ys for x in xs]

        if self.include_baseline:
            self.centers.append(None)   # <- baseline sample

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        center = self.centers[idx]

        w = h = self.image_size
        bg_rgba = Image.new("RGBA", (w, h), self.bg_color + (255,))
        bg_rgba.paste(self.base, (0, 0), mask=self.base)

        if center is not None:
            cx, cy = center
            draw = ImageDraw.Draw(bg_rgba)
            r = self.circle_radius
            bbox = (
                int(round(cx - r)),
                int(round(cy - r)),
                int(round(cx + r)),
                int(round(cy + r)),
            )
            draw.ellipse(bbox, outline=(255, 0, 0, 255), width=self.line_width)

        if self.output_size is not None:
            bg_rgba = bg_rgba.resize(self.output_size, resample=Image.BICUBIC)
        return bg_rgba.convert("RGB")


def run_vlm_sampling_red_circle(
    model: Union[LlavaForConditionalGeneration, LlavaNextForConditionalGeneration],
    processor: Union[LlavaProcessor, LlavaNextProcessor],
    dataset: Dataset,
    prompt: str,
    sample_n: int = 10,
    batch_size: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
    use_logit: bool = True,
    top_k: int = 100,
    prompt_format: str = None,
    prompt_prefix: str = None,
    extra_tokens: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Run a VLM on every image of `dataset` with the processor's chat template.

    Args:
        model: the loaded VLM
        processor: its AutoProcessor
        dataset: a Dataset returning PIL images
        prompt: the user text prompt
        batch_size: images per forward pass
        device: torch.device (auto-detected when None)

    Returns:
        one record per image (generated texts and/or first-token probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    
    if prompt_format is None:
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image'}
                ],
            },
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        text_prompt = prompt_format.format(prompt=prompt)

    if prompt_prefix is not None:
        text_prompt = text_prompt + prompt_prefix


    results: List[Dict[str, object]] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
  # Process batch by batch; images within a batch are generated in parallel.
        for batch_idx, batch in enumerate(loader):
            images = batch['pixel_values']
  # angles of this batch
            start = batch_idx * batch_size
            batch_centers = dataset.centers[start: start + len(images)]
  # one output list per image
            outputs_per_img = {center: [] for center in batch_centers}

  # sample the whole batch at once
            texts = [text_prompt] * len(images)
            inputs = processor(
                images=images,
                text=texts,
                return_tensors='pt'
            ).to(device)

            if use_logit:
                tok = processor.tokenizer
                extra_tokens = extra_tokens or []
                vocab = tok.get_vocab()
                unk_id = getattr(tok, "unk_token_id", None)
                unk_tok = getattr(tok, "unk_token", None)

                extra_tok_ids: List[int] = []
                extra_tok_strs: List[str] = []
                for t in extra_tokens:
                    tid = vocab.get(t, None)
                    if tid is None:
                        # fallback: convert_tokens_to_ids may return unk_id for missing
                        cand = tok.convert_tokens_to_ids(t)
                        if isinstance(cand, int) and cand != unk_id:
                            tid = cand
                        elif unk_id is not None and cand == unk_id and t == unk_tok:
                            tid = cand  # allow explicit <unk>
                        else:
                            tid = None
                    if tid is not None:
                        extra_tok_ids.append(int(tid))
                        extra_tok_strs.append(t)

                gen_out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
                first_token_logits = gen_out.scores[0]  # (B, V)
                probs_all = torch.softmax(first_token_logits, dim=-1)  # (B, V)

                top_values, top_indices = first_token_logits.topk(k=top_k, dim=-1)  # (B, K)
                top_probs = probs_all.gather(dim=-1, index=top_indices)              # (B, K)

                for row_i, center in enumerate(batch_centers):
                    idx_list = top_indices[row_i].detach().cpu().tolist()
                    tok_list = tok.convert_ids_to_tokens(idx_list)
                    logit_list = top_values[row_i].detach().cpu().tolist()
                    prob_list  = top_probs[row_i].detach().cpu().tolist()

                    # Append extra tokens if not already in the top-k.
                    present_ids = set(idx_list)
                    for t_str, t_id in zip(extra_tok_strs, extra_tok_ids):
                        if t_id in present_ids:
                            continue
                        present_ids.add(t_id)
                        tok_list.append(t_str)
                        idx_list.append(t_id)
                        logit_list.append(float(first_token_logits[row_i, t_id].detach().cpu()))
                        prob_list.append(float(probs_all[row_i, t_id].detach().cpu()))

                    outputs_per_img[center].append({
                        "tokens": tok_list,
                        "indices": idx_list,
                        "logits":  logit_list,
                        "probs":   prob_list,
                    })
            else:
  # loop over the samples
                for n in range(sample_n):
                    torch.manual_seed(int(seed + n))

                    gen_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=len(images),
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                    gen_ids = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
                    ]
                    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

  # distribute to the images of the batch
                    for center, text_out in zip(batch_centers, decoded):
                        outputs_per_img[center].append(text_out)

                    del gen_ids
                    torch.cuda.empty_cache()

  # collect the results
            for center in batch_centers:
                results.append({'center': center, 'output': outputs_per_img[center]})
            
            del inputs
            torch.cuda.empty_cache()
    
    return results


import json
from typing import List, Dict, Any, Optional, Tuple

def _upsample_grid_bilinear(
    grid_values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    grid_values: shape (len(ys), len(xs))  (y, x)
    xs, ys: sample coordinates in image space
    width, height: target image size
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ny, nx = grid_values.shape

    x_new = np.arange(width, dtype=float)
    y_new = np.arange(height, dtype=float)

    # 1) interpolate along x for each y
    tmp = np.full((ny, width), np.nan, dtype=float)
    for iy in range(ny):
        row = grid_values[iy]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        x_valid = xs[valid]
        z_valid = row[valid]
        if len(x_valid) == 1:
            tmp[iy, :] = z_valid[0]
        else:
            tmp[iy, :] = np.interp(
                x_new, x_valid, z_valid, left=z_valid[0], right=z_valid[-1]
            )

    # 2) interpolate along y for each x
    out = np.full((height, width), np.nan, dtype=float)
    for ix in range(width):
        col = tmp[:, ix]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        y_valid = ys[valid]
        z_valid = col[valid]
        if len(y_valid) == 1:
            out[:, ix] = z_valid[0]
        else:
            out[:, ix] = np.interp(
                y_new, y_valid, z_valid, left=z_valid[0], right=z_valid[-1]
            )

    return out

def plot_heatmaps(
    input_path: str,
    image_path: str,
    group1_tokens: List[str],
    group2_tokens: List[str],
    use_logits: bool = True,
    figsize: Tuple[int, int] = (18, 6),
    alpha: float = 0.6,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    subtract_baseline: bool = True,
    group1_label: str = "Rabbit",
    group2_label: str = "Bird",
) -> None:
    """
    Read the JSON produced by run_vlm_sampling_red_circle and create
    three heatmaps overlaid on the original image:

      1. group1 values (or group1 - baseline if subtract_baseline=True)
      2. group2 values (or group2 - baseline if subtract_baseline=True)
      3. difference:
         - raw: group1 - group2
         - baseline-subtracted:
           (group1 - group2) - (baseline_group1 - baseline_group2)

    Assumes the dataset may contain one extra baseline sample with:
        "center": null
    i.e. Python None serialized to JSON.

    Args:
        input_path: Path to JSON file from run_vlm_sampling_red_circle.
        image_path: Path to the original background image.
        group1_tokens: Tokens for group 1.
        group2_tokens: Tokens for group 2.
        use_logits: If True, use "logits"; otherwise use "probs".
        figsize: Figure size.
        alpha: Heatmap overlay opacity.
        save_path: If provided, save figure here.
        title: Optional figure title.
        subtract_baseline: Whether to subtract the no-red-circle baseline.
    """
    # --------------------------------------------------------------
    # 1. Load JSON and collect raw values at each center
    # --------------------------------------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    centers = []
    g1_vals_raw = []
    g2_vals_raw = []

    baseline_g1 = None
    baseline_g2 = None

    for item in data:
        center = item.get("center", "missing")
        outputs = item.get("output", [])
        if center == "missing" or not outputs:
            continue

        out = outputs[0]
        tokens = out.get("tokens", [])
        vals_logits = out.get("logits", [])
        vals_probs = out.get("probs", [])

        if not tokens:
            continue

        # Choose which values to use
        if use_logits and len(vals_logits) == len(tokens):
            vals = vals_logits
        elif (not use_logits) and len(vals_probs) == len(tokens):
            vals = vals_probs
        elif len(vals_logits) == len(tokens):
            vals = vals_logits
        elif len(vals_probs) == len(tokens):
            vals = vals_probs
        else:
            continue

        token_to_val = {t: float(v) for t, v in zip(tokens, vals)}

        def sum_group(group_tokens: List[str]) -> float:
            if not group_tokens:
                return np.nan
            arr = np.array(
                [token_to_val.get(t, np.nan) for t in group_tokens],
                dtype=float,
            )
            if np.all(np.isnan(arr)):
                return np.nan
            return float(np.nansum(arr))

        v1 = sum_group(group1_tokens)
        v2 = sum_group(group2_tokens)

        # center=None is the baseline sample
        if center is None:
            baseline_g1 = v1
            baseline_g2 = v2
            continue

        cx, cy = center
        centers.append((cx, cy))
        g1_vals_raw.append(v1)
        g2_vals_raw.append(v2)

    if not centers:
        raise ValueError("No usable non-baseline centers found in JSON.")

    centers = np.array(centers, dtype=float)
    g1_vals_raw = np.array(g1_vals_raw, dtype=float)
    g2_vals_raw = np.array(g2_vals_raw, dtype=float)

    # --------------------------------------------------------------
    # 2. Apply baseline subtraction if requested
    # --------------------------------------------------------------
    if subtract_baseline:
        if baseline_g1 is None or baseline_g2 is None:
            raise ValueError(
                "subtract_baseline=True but no baseline sample (center=None) was found."
            )

        g1_vals = g1_vals_raw - baseline_g1
        g2_vals = g2_vals_raw - baseline_g2
        baseline_diff = baseline_g1 - baseline_g2
        diff_vals = (g1_vals_raw - g2_vals_raw) - baseline_diff
    else:
        g1_vals = g1_vals_raw
        g2_vals = g2_vals_raw
        diff_vals = g1_vals_raw - g2_vals_raw

    # --------------------------------------------------------------
    # 3. Load base image and keep only in-bounds centers
    # --------------------------------------------------------------
    base_img = Image.open(image_path).convert("RGB")
    w, h = base_img.size

    xs_img = centers[:, 0]
    ys_img = centers[:, 1]

    inside = (
        (xs_img >= 0) & (xs_img < w) &
        (ys_img >= 0) & (ys_img < h)
    )

    xs_img = xs_img[inside].astype(int)
    ys_img = ys_img[inside].astype(int)
    g1_vals = g1_vals[inside]
    g2_vals = g2_vals[inside]
    diff_vals = diff_vals[inside]

    if xs_img.size == 0:
        raise ValueError(
            "No centers are inside the original image region. "
            "Check that the JSON 'center' values match the dataset geometry."
        )

    # --------------------------------------------------------------
    # 4. Put values on a 2D grid
    # --------------------------------------------------------------
    xs_unique = np.sort(np.unique(xs_img))
    ys_unique = np.sort(np.unique(ys_img))

    nx = len(xs_unique)
    ny = len(ys_unique)

    x_to_ix = {x: i for i, x in enumerate(xs_unique)}
    y_to_iy = {y: i for i, y in enumerate(ys_unique)}

    grid_g1 = np.full((ny, nx), np.nan, dtype=float)
    grid_g2 = np.full((ny, nx), np.nan, dtype=float)
    grid_diff = np.full((ny, nx), np.nan, dtype=float)

    for x, y, v1, v2, vd in zip(xs_img, ys_img, g1_vals, g2_vals, diff_vals):
        ix = x_to_ix[x]
        iy = y_to_iy[y]
        grid_g1[iy, ix] = v1
        grid_g2[iy, ix] = v2
        grid_diff[iy, ix] = vd

    # --------------------------------------------------------------
    # 5. Upsample to full image size
    # --------------------------------------------------------------
    heat_g1 = _upsample_grid_bilinear(grid_g1, xs_unique, ys_unique, w, h)
    heat_g2 = _upsample_grid_bilinear(grid_g2, xs_unique, ys_unique, w, h)
    heat_diff = _upsample_grid_bilinear(grid_diff, xs_unique, ys_unique, w, h)

    # --------------------------------------------------------------
    # 6. Compute color scales
    # --------------------------------------------------------------
    valid12 = np.concatenate([
        heat_g1[~np.isnan(heat_g1)],
        heat_g2[~np.isnan(heat_g2)],
    ])
    if valid12.size == 0:
        raise ValueError("All values for group1 and group2 are NaN.")

    vmin12 = float(valid12.min())
    vmax12 = float(valid12.max())
    if vmin12 == vmax12:
        eps = max(1e-6, abs(vmin12) * 0.1)
        vmin12 -= eps
        vmax12 += eps

    valid_diff = heat_diff[~np.isnan(heat_diff)]
    if valid_diff.size == 0:
        vmin_diff, vmax_diff = -1.0, 1.0
    else:
        max_abs_diff = float(np.max(np.abs(valid_diff)))
        if max_abs_diff == 0.0:
            max_abs_diff = 1.0
        vmin_diff = -max_abs_diff
        vmax_diff = max_abs_diff

    # --------------------------------------------------------------
    # 7. Plot
    # --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    heatmaps = [heat_g1, heat_g2, heat_diff]

    if subtract_baseline:
        ax_titles = [
            f"Δ({group1_label})",
            f"Δ({group2_label})",
            f"Δ({group1_label} - {group2_label})",
        ]
    else:
        ax_titles = [
            group1_label,
            group2_label,
            f"{group1_label} - {group2_label}",
        ]

    cmaps = ["Reds", "Blues", "RdBu_r"]
    
    if subtract_baseline:
        cbar_labels = [
            "Logit change from baseline" if use_logits else "Probability change from baseline",
            "Logit change from baseline" if use_logits else "Probability change from baseline",
            "Change in logit difference from baseline" if use_logits else "Change in probability difference from baseline",
        ]
    else:
        cbar_labels = [
            "Logit" if use_logits else "Probability",
            "Logit" if use_logits else "Probability",
            "Logit difference" if use_logits else "Probability difference",
        ]

    for i, (ax, heat, ax_title) in enumerate(zip(axes, heatmaps, ax_titles)):
        canvas_arr = np.array(base_img)
        ax.imshow(canvas_arr)
        hm = np.ma.masked_invalid(heat)

        if i < 2:
            im = ax.imshow(
                hm,
                alpha=alpha,
                cmap=cmaps[i],
                vmin=vmin12,
                vmax=vmax12,
            )
        else:
            im = ax.imshow(
                hm,
                alpha=alpha,
                cmap=cmaps[i],
                vmin=vmin_diff,
                vmax=vmax_diff,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ax_title)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_labels[i])

    if title:
        fig.suptitle(title)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def float_tag(x: float, ndigits: int = 3) -> str:
    """
    Convert float to a filename-safe tag.
    Examples:
      0.071428 -> '0p071'
      0.5      -> '0p500'
      -0.125   -> 'm0p125'
    """
    s = f"{x:.{ndigits}f}"
    s = s.replace("-", "m").replace(".", "p")
    return s


# logitlens
def _rmsnorm_apply_with_ref(norm_module, x, x_ref):
    eps = getattr(norm_module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm_module, "eps", 1e-6)
    denom = torch.sqrt(x_ref.float().pow(2).mean(dim=-1, keepdim=True) + eps)  # [P,1]
    return (x / denom.to(dtype=x.dtype)) * norm_module.weight.to(dtype=x.dtype)

def _layernorm_apply_with_ref(norm_module, x, x_ref):
    eps = getattr(norm_module, "eps", 1e-5)
    mu = x_ref.mean(dim=-1, keepdim=True)
    var = (x_ref - mu).pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    y = (x - mu) * inv
    if hasattr(norm_module, "weight") and norm_module.weight is not None:
        y = y * norm_module.weight.to(dtype=x.dtype)
    if hasattr(norm_module, "bias") and norm_module.bias is not None:
        y = y + norm_module.bias.to(dtype=x.dtype)
    return y

def _apply_final_norm(norm_module, x, ln_mode: str, x_ref=None):
    if ln_mode == "none":
        return x
    if ln_mode == "independent":
        return norm_module(x)
    if ln_mode == "cached":
        if x_ref is None:
            raise ValueError("ln_mode='cached' requires ln_ref/x_ref.")
        is_rms = (
            hasattr(norm_module, "variance_epsilon")
            and hasattr(norm_module, "weight")
            and not hasattr(norm_module, "bias")
        )
        if is_rms or norm_module.__class__.__name__.lower().endswith("rmsnorm"):
            return _rmsnorm_apply_with_ref(norm_module, x, x_ref)
        else:
            return _layernorm_apply_with_ref(norm_module, x, x_ref)
    raise ValueError(f"Unknown ln_mode: {ln_mode}")

def _fix_pos(pos_list, S: int, device):
    # pos_list: list[int] (may contain negatives like -1)
    pos = torch.tensor(pos_list, dtype=torch.long)   # keep on CPU for checks
    pos = pos.clone()
    pos[pos < 0] += S
    if (pos < 0).any() or (pos >= S).any():
        bad = pos[(pos < 0) | (pos >= S)].tolist()
        raise IndexError(f"pos out of range for S={S}. Bad indices (after fixing): {bad}")
    return pos.to(device)

def logitlens(
    model,
    hidden_states,
    tokens_of_interest,
    pos,  # list[int]
    softmax=False,
    head_out=False,
    device="cpu",
    save_path=None,
    ln_mode: str = "independent",   # ["independent", "cached", "none"]
    ln_ref=None,               # required if ln_mode == "cached"
):
    """
    DLA-friendly logitlens by default (cached final norm stats).

    hidden_states:
      - not head_out: list/tuple length L, each [S,D] (or np arrays)
      - head_out: list/tuple length L, each [S,H,D] (or np arrays)
    ln_ref:
      - if ln_mode='cached': either [S,D] (single ref for all layers) OR [L,S,D] (per-layer ref)
        Typical: cache["resid_post"][-1]  (final full residual, pre-final-norm)
    """
    interesting_token_ids = torch.tensor(list(tokens_of_interest.values()), device=device, dtype=torch.long)

    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    # prepare ln_ref tensor if needed
    ln_ref_t = None
    if ln_mode == "cached":
        if ln_ref is None:
            raise ValueError("ln_mode='cached' is default, so please pass ln_ref (e.g., cache['resid_post'][-1]).")
        ln_ref_t = torch.from_numpy(ln_ref) if isinstance(ln_ref, np.ndarray) else ln_ref
        ln_ref_t = ln_ref_t.to(device)

    if not head_out:
        L = len(hidden_states)
        out = torch.zeros((L, len(pos), len(tokens_of_interest)), device=device, dtype=torch.float32)

        for l in range(L):
            x = hidden_states[l]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(device)                     # [S,D]
            pos_t = _fix_pos(pos, x.shape[0], device)
            x = x.index_select(0, pos_t)          # [P,D]

            if ln_mode == "cached":
                ref = ln_ref_t[l] if (ln_ref_t.ndim == 3 and ln_ref_t.shape[0] == L) else ln_ref_t
                if ref.ndim != 2:
                    raise ValueError(f"ln_ref must be [S,D] or [L,S,D], got {tuple(ref.shape)}")
                ref = ref.index_select(0, pos_t)  # [P,D]
                x_norm = _apply_final_norm(norm, x, "cached", x_ref=ref)
            else:
                x_norm = _apply_final_norm(norm, x, ln_mode)

            logits = lm_head(x_norm)  # [P,V]
            if softmax:
                logits = torch.softmax(logits, dim=-1)

            out[l] = logits.index_select(1, interesting_token_ids).float()

        out_np = out.detach().cpu().numpy()
        if save_path is not None:
            np.save(save_path, out_np)
        return out_np

    else:
        L, H, S, D = hidden_states.shape
        out = torch.zeros((L, H, len(pos), len(tokens_of_interest)), device=device, dtype=torch.float32)

        for l in range(L):
            xh = hidden_states[l]
            if isinstance(xh, np.ndarray):
                xh = torch.from_numpy(xh)
            xh = xh.to(device)                  # [H,S,D]
            pos_t = _fix_pos(pos, xh.shape[1], device)
            xh = xh.index_select(1, pos_t)       # [H,P,D]

            if ln_mode == "cached":
                ref = ln_ref_t[l] if (ln_ref_t.ndim == 3 and ln_ref_t.shape[0] == L) else ln_ref_t
                if ref.ndim != 2:
                    raise ValueError(f"ln_ref must be [S,D] or [L,S,D], got {tuple(ref.shape)}")
                ref = ref.index_select(0, pos_t)  # [P,D]
            else:
                ref = None

            for h in range(H):
                x = xh[h, :, :]  # [P,D]
                if ln_mode == "cached":
                    x_norm = _apply_final_norm(norm, x, "cached", x_ref=ref)
                else:
                    x_norm = _apply_final_norm(norm, x, ln_mode)

                logits = lm_head(x_norm)
                if softmax:
                    logits = torch.softmax(logits, dim=-1)
                out[l, h, :, :] = logits.index_select(1, interesting_token_ids).float()

        out_np = out.detach().cpu().numpy()
        if save_path is not None:
            np.save(save_path, out_np)
        return out_np

def base_view_image_positions(processor, model, text: str, image, n_base: int = 576):
    """Sequence positions of the BASE-VIEW image tokens for (text, image).

    Positions are read off the processor-expanded input_ids (== image_token_index),
    so they are correct for every prompt template (LLaVA-1.5: 5 prefix tokens,
    Vicuna-1.6: 35, Llama3-NEXT: 46, ...). For AnyRes models HF prepends the
    576-token global view before the high-res tiles, so the first n_base image
    positions are the base view; llava-1.5 has exactly n_base.
    """
    inputs = processor(text=text, images=image, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    img_id = int(model.config.image_token_index)
    pos = [i for i, t in enumerate(ids) if t == img_id]
    if len(pos) < n_base:
        raise ValueError(f"only {len(pos)} image tokens found (< {n_base}); unexpanded input_ids?")
    return pos[:n_base]


def dominant_patch_map_from_logitlens(
    logitlens_probs: np.ndarray,          # [L, P, T] from logitlens(..., softmax=True)
    tokens_of_interest: dict[str, int],   # keeps token ordering via keys()
    threshold: float = 0.1,
    img_pos: list[int] | np.ndarray | None = None,
    image_size: int = 336,
    patch_size: int = 14,
    pre_img_tokens: int = 5,              # your convention in show_logitlens_of_image_tokens
):
    """
    Returns:
      dominant_tokens: list[str|None] length P_img
      dominant_values: np.ndarray float (P_img,) with None-masked as np.nan (or you can return Python None list)
      per_token_max:   np.ndarray float (P_img, T)  (max over layers for each token)
      best_idx:        np.ndarray int   (P_img,)     (argmax over tokens)
    """
    assert logitlens_probs.ndim == 3, f"Expected [L,P,T], got {logitlens_probs.shape}"
    L, P, T = logitlens_probs.shape

    token_names = list(tokens_of_interest.keys())
    assert T == len(token_names), f"T mismatch: probs has T={T}, tokens_of_interest has {len(token_names)}"

    # Decide which positions are "image tokens"
    if img_pos is None:
        img_token_count = (image_size // patch_size) ** 2
        img_start = pre_img_tokens
        img_end = min(img_start + img_token_count, P)
        img_pos = np.arange(img_start, img_end)
    else:
        img_pos = np.asarray(img_pos)

    # [L, P_img, T]
    img_probs = logitlens_probs[:, img_pos, :].astype(np.float32)

    # For each (p_img, t): max over layers
    # [P_img, T]
    dominant_patch_map = img_probs.max(axis=0)
    below_threshold_map = dominant_patch_map < threshold
    dominant_patch_map[below_threshold_map] = 0

    return dominant_patch_map

def show_dominant_patch_map(
    dominant_patch_map: np.ndarray,
    image: Image.Image,
    tokens_of_interest: dict[str, int],
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    x_y_pairs: list[tuple[int, int]] = [[0, 1]],
):
    for idx, token in enumerate(list(tokens_of_interest.keys())):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
        dominant_patch_map_up = np.kron(dominant_patch_map[..., idx].reshape(image_size//patch_size, image_size//patch_size), np.ones((patch_size, patch_size)))
        im = ax.imshow(dominant_patch_map_up, cmap="Blues", vmin=0, vmax=1, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(
            im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"max prob: {token}"
        )
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{token}.png"), bbox_inches='tight', pad_inches=0.05)

    for x_idx, y_idx in x_y_pairs:
        x_token = list(tokens_of_interest.keys())[x_idx]
        y_token = list(tokens_of_interest.keys())[y_idx]
        fig, ax = plt.subplots(figsize=(10, 10))
        dominant_patch_map_diff = dominant_patch_map[..., x_idx] - dominant_patch_map[..., y_idx]

        ax.imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
        dominant_patch_map_up = np.kron(dominant_patch_map_diff.reshape(image_size//patch_size, image_size//patch_size), np.ones((patch_size, patch_size)))
        im = ax.imshow(dominant_patch_map_up, cmap="RdBu", vmin=-1, vmax=1, alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(
            im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"max prob diff: {x_token} - {y_token}"
        )
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{x_token}_{y_token}.png"), bbox_inches='tight', pad_inches=0.05)
        plt.close()

# ablation (zero/resample) and visualization

from typing import Optional, Literal, Tuple

def _get_fill_value(dtype: torch.dtype, strategy: Literal["-inf", "finfo_min"]):
    if strategy == "-inf":
        # literal -inf (preferred)
        return float("-inf")
    elif strategy == "finfo_min":
        # safer fallback for some fp16 kernels
        return torch.finfo(dtype).min
    else:
        raise ValueError(f"Unknown knockout_value strategy: {strategy}")


def _ensure_4d_additive_mask_from_2d(
    attn_mask_2d: torch.Tensor,  # [B, S], usually 1 for keep, 0 for pad
    hidden_dtype: torch.dtype,
    fill_value,
) -> torch.Tensor:
    """
    Build a standard additive causal+padding mask of shape [B, 1, S, S].
    0 means allowed, fill_value means masked.
    """
    assert attn_mask_2d.dim() == 2
    B, S = attn_mask_2d.shape
    device = attn_mask_2d.device

    # causal mask: mask future positions (j > i)
    causal = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)  # [S, S]
    out = torch.zeros((B, 1, S, S), device=device, dtype=hidden_dtype)
    out = out.masked_fill(causal[None, None, :, :], fill_value)

    # padding mask: mask pad tokens for all queries
    # HF-style attention_mask is often 1=keep, 0=pad
    pad = (attn_mask_2d == 0)
    if pad.any():
        out = out.masked_fill(pad[:, None, None, :], fill_value)

    return out

from typing import Optional, Literal, Tuple, Dict
import torch

def _infer_img_token_id(processor, model, img_token_id: Optional[int]) -> int:
    if img_token_id is not None:
        return img_token_id
    # try config first (common for LLaVA)
    v = getattr(model.config, "image_token_index", None)
    if v is None and hasattr(model, "language_model"):
        v = getattr(model.language_model.config, "image_token_index", None)
    if v is not None:
        return int(v)
    # fallback: tokenizer
    tok = processor.tokenizer
    v = tok.convert_tokens_to_ids("<image>")
    if v == tok.unk_token_id:
        raise ValueError("Could not infer img_token_id: '<image>' is not a known token.")
    return int(v)

def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    """Return start index of needle in haystack, or -1."""
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return -1
    # brute force (S is small enough)
    H = haystack.tolist()
    N = needle.tolist()
    n = len(N)
    for i in range(len(H) - n + 1):
        if H[i:i+n] == N:
            return i
    return -1

def _compute_token_groups(
    input_ids_1d: torch.Tensor,     # [S]
    tokenizer,
    img_token_id: int,
    assistant_marker: str = "ASSISTANT:",
) -> Dict[str, torch.Tensor]:
    """
    Returns dict of 1D LongTensors (positions) on same device:
      image, query, gen, gen_but_last, last
    Heuristic for gen: tokens at/after the first occurrence of tokenized assistant_marker.
    """
    assert input_ids_1d.dim() == 1
    device = input_ids_1d.device
    S = int(input_ids_1d.numel())
    all_pos = torch.arange(S, device=device, dtype=torch.long)
    last_pos = S - 1

    image_pos = (input_ids_1d == img_token_id).nonzero(as_tuple=True)[0].to(torch.long)
    # assistant marker boundary
    marker_ids = torch.tensor(tokenizer.encode(assistant_marker, add_special_tokens=False), device=device, dtype=torch.long)
    start = _find_subsequence(input_ids_1d, marker_ids)
    if start == -1:
        gen_start = S  # no assistant segment found
    else:
        gen_start = start + int(marker_ids.numel())

    gen_pos = all_pos[gen_start:]                 # could include last
    gen_but_last = gen_pos[gen_pos != last_pos]   # exclude last
    # "query" = everything before gen_start except image tokens
    is_img = torch.zeros(S, dtype=torch.bool, device=device)
    if image_pos.numel() > 0:
        is_img[image_pos] = True
    query_pos = all_pos[:gen_start]
    query_pos = query_pos[~is_img[query_pos]]

    groups = {
        "image": image_pos,
        "query": query_pos,
        "gen": gen_pos,
        "gen_but_last": gen_but_last,
        "last": torch.tensor([last_pos], device=device, dtype=torch.long),
    }
    return groups

def _select_query_positions(S: int, which: Optional[Literal["last", "all"]], device) -> torch.Tensor:
    if which is None:
        return torch.tensor([], device=device, dtype=torch.long)
    if which == "last":
        return torch.tensor([S - 1], device=device, dtype=torch.long)
    if which == "all":
        return torch.arange(S, device=device, dtype=torch.long)
    raise ValueError(f"Unknown attention_knockout_query: {which}")

def _positions_to_zero_or_mask(
    S: int,
    key_pos: torch.Tensor,
    mode: Literal["drop", "keep_only"],
    device,
) -> torch.Tensor:
    """
    Returns positions to be masked/zeroed (keys).
    - drop: mask key_pos
    - keep_only: mask complement of key_pos
    """
    if mode == "drop":
        return key_pos
    elif mode == "keep_only":
        if key_pos.numel() == 0:
            # pre_softmax would create all-masked rows -> NaNs; post_softmax is okay (would zero everything)
            return torch.arange(S, device=device, dtype=torch.long)
        mask = torch.ones(S, dtype=torch.bool, device=device)
        mask[key_pos] = False
        return torch.arange(S, device=device, dtype=torch.long)[mask]
    else:
        raise ValueError(f"Unknown attention_knockout_mode: {mode}")


from typing import Optional, Literal, Tuple, Dict
import os
import torch

def get_cache(
    model,
    processor,
    image,
    prompt,
    device="cpu",
    save_dir=None,
    attention_knockout_layer: Optional[int] = None,
    attention_knockout_key: Optional[Literal["image", "query", "gen_but_last", "last"]] = None,
    attention_knockout_query: Optional[Literal["last", "all"]] = None,
    attention_knockout_mode: Optional[Literal["drop", "keep_only"]] = "drop",
    attention_knockout_stage: Optional[Literal["pre_softmax", "post_softmax"]] = "pre_softmax",
    img_token_id: int = 32000,
    knockout_value: Literal["-inf", "finfo_min"] = "-inf",
    cache_keys: Optional[List[str]] = None,   # NEW
):
    """
    cache_keys:
      None -> keep old behavior (build/save all cache tensors)
      otherwise -> only build/save requested top-level keys, e.g.
                   ["embed", "resid_post"] or ["resid_post"]
    """
    all_cache_keys = [
        "resid_post", "attn_map", "attn_out", "mlp_out", "resid_mid",
        "attn_x_value", "head_out", "resid", "embed", "vproj_out"
    ]
    requested = set(all_cache_keys if cache_keys is None else cache_keys)


    # load only requested keys if available
    if save_dir is not None:
        can_load = all(os.path.exists(os.path.join(save_dir, f"{k}.pt")) for k in requested)
        if can_load:
            cache = {}
            for key in requested:
                cache[key] = torch.load(os.path.join(save_dir, f"{key}.pt"))
            return cache

    llm = model.language_model.model
    L = len(llm.layers)
    cache = {}

    # what do we actually need to capture?
    need_embed = ("embed" in requested)
    need_resid_post = ("resid_post" in requested) or ("resid" in requested)
    need_resid_mid = ("resid_mid" in requested) or ("resid" in requested)
    need_attn_out = ("attn_out" in requested)
    need_mlp_out = ("mlp_out" in requested)
    need_attn_map = ("attn_map" in requested) or ("attn_x_value" in requested)
    need_vproj_out = ("vproj_out" in requested) or ("attn_x_value" in requested)
    need_head_out = ("head_out" in requested)

    # preprocess
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids_1d = inputs["input_ids"][0]
    S = int(input_ids_1d.numel())

    img_token_id_eff = _infer_img_token_id(processor, model, img_token_id)

    # preallocate only what is needed
    attn_out = [None] * L if need_attn_out else None
    mlp_out = [None] * L if need_mlp_out else None
    resid_mid = [None] * L if need_resid_mid else None
    resid_post = [None] * L if need_resid_post else None
    resid_pre0 = [None] if need_resid_post else None
    flat_heads = [None] * L if need_head_out else None
    vproj_out = [None] * L if (need_vproj_out or (attention_knockout_stage == "post_softmax" and attention_knockout_layer is not None)) else None
    embed = [None] if need_embed else None

    # capture embeddings
    if need_embed:
        def _embed_capture_hook(module, args, kwargs):
            e = None
            if kwargs is not None:
                e = kwargs.get("inputs_embeds", None)
                if e is None:
                    input_ids = kwargs.get("input_ids", None)
                    if input_ids is not None and hasattr(module, "embed_tokens"):
                        e = module.embed_tokens(input_ids)
            if e is not None:
                embed[0] = e[0].detach()
            return (args, kwargs)

        embed_handle = llm.register_forward_pre_hook(_embed_capture_hook, with_kwargs=True)
    else:
        embed_handle = None

    # -------------------------
    # knockout hooks (unchanged logic)
    # -------------------------
    knockout_handles = []
    patched_attn = None
    patched_forward_orig = None

    do_knockout = (
        attention_knockout_layer is not None
        and attention_knockout_key is not None
        and attention_knockout_query is not None
    )

    if do_knockout:
        if not (0 <= attention_knockout_layer < L):
            raise ValueError(f"attention_knockout_layer out of range: {attention_knockout_layer}")
        if attention_knockout_mode is None or attention_knockout_stage is None:
            raise ValueError("If attention_knockout_layer is set, mode and stage must be set too.")

        groups = _compute_token_groups(
            input_ids_1d=input_ids_1d,
            tokenizer=processor.tokenizer,
            img_token_id=img_token_id_eff,
            assistant_marker="ASSISTANT:",
        )
        key_pos = groups[attention_knockout_key]
        q_pos = _select_query_positions(S, attention_knockout_query, device=input_ids_1d.device)
        keys_to_zero_or_mask = _positions_to_zero_or_mask(
            S=S,
            key_pos=key_pos,
            mode=attention_knockout_mode,
            device=input_ids_1d.device,
        )

        if q_pos.numel() == 0 or keys_to_zero_or_mask.numel() == 0:
            do_knockout = False

    if do_knockout and attention_knockout_stage == "pre_softmax":
        blk = llm.layers[attention_knockout_layer]
        fill = _get_fill_value(dtype=model.dtype, strategy=knockout_value)

        def _apply_pre_softmax_mask(args, kwargs):
            if kwargs is None:
                return kwargs
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is None:
                return kwargs

            if attn_mask.dim() == 4:
                mask = attn_mask.clone()
                if mask.dtype == torch.bool:
                    mask[:, :, q_pos[:, None], keys_to_zero_or_mask[None, :]] = True
                else:
                    fv = torch.tensor(fill, device=mask.device, dtype=mask.dtype) if not isinstance(fill, float) else fill
                    mask[:, :, q_pos[:, None], keys_to_zero_or_mask[None, :]] = fv
                kwargs["attention_mask"] = mask
                return kwargs

            if attn_mask.dim() == 2:
                hidden_states = args[0]
                base = _ensure_4d_additive_mask_from_2d(
                    attn_mask_2d=attn_mask,
                    hidden_dtype=hidden_states.dtype,
                    fill_value=fill,
                )
                base[:, :, q_pos[:, None], keys_to_zero_or_mask[None, :]] = fill
                kwargs["attention_mask"] = base
                return kwargs

            return kwargs

        def _pre_hook(module, args, kwargs):
            kwargs = _apply_pre_softmax_mask(args, kwargs)
            return (args, kwargs)

        try:
            knockout_handles.append(blk.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        except TypeError:
            patched_attn = blk.self_attn
            patched_forward_orig = patched_attn.forward

            def _forward_patched(*args, **kwargs):
                kwargs = _apply_pre_softmax_mask(args, kwargs)
                return patched_forward_orig(*args, **kwargs)

            patched_attn.forward = _forward_patched

    if do_knockout and attention_knockout_stage == "post_softmax":
        blk = llm.layers[attention_knockout_layer]

        def _vproj_hook_knockout(module, inp, out, layer_idx=attention_knockout_layer):
            vproj_out[layer_idx] = out.detach()

        knockout_handles.append(blk.self_attn.v_proj.register_forward_hook(_vproj_hook_knockout))

        def _self_attn_rewrite_hook(module, inp, out, layer_idx=attention_knockout_layer):
            if not isinstance(out, tuple) or len(out) < 2:
                return out
            attn_weights = out[1]
            rest = out[2:] if len(out) > 2 else ()

            if attn_weights is None:
                raise RuntimeError(
                    "post_softmax knockout requested but attn_weights is None. "
                    "You likely need eager attention while output_attentions=True."
                )
            if vproj_out[layer_idx] is None:
                raise RuntimeError("v_proj output was not captured; cannot do post_softmax rewrite.")

            B, H, q_len, kv_len = attn_weights.shape
            v = vproj_out[layer_idx]
            if v.shape[1] != kv_len:
                raise RuntimeError(f"kv_len mismatch: attn_weights kv_len={kv_len}, v_proj len={v.shape[1]}")

            hd = module.o_proj.weight.shape[1] // H
            if v.shape[-1] % hd != 0:
                raise RuntimeError(f"v dim {v.shape[-1]} not divisible by head_dim={hd}")
            H_kv = v.shape[-1] // hd
            value_states = v.view(B, kv_len, H_kv, hd).transpose(1, 2).contiguous()
            if H_kv != H:
                # Grouped-query attention (mistral, llama3): expand shared value
                # heads per query head (same as the model's internal repeat_kv).
                value_states = value_states.repeat_interleave(H // H_kv, dim=1)

            w = attn_weights.clone()
            w[:, :, q_pos[:, None], keys_to_zero_or_mask[None, :]] = 0.0

            out_heads = torch.matmul(w, value_states)
            out_concat = out_heads.transpose(1, 2).contiguous().view(B, q_len, H * hd)
            new_attn_output = module.o_proj(out_concat)

            if len(out) == 2:
                return (new_attn_output, w)
            else:
                return (new_attn_output, w, *rest)

        knockout_handles.append(blk.self_attn.register_forward_hook(_self_attn_rewrite_hook))

    # -------------------------
    # cache hooks
    # -------------------------
    handles = []
    for layer_idx, blk in enumerate(llm.layers):
        if need_vproj_out:
            def vproj_hook(module, inp, out, layer_idx=layer_idx):
                vproj_out[layer_idx] = out.detach()
            handles.append(blk.self_attn.v_proj.register_forward_hook(vproj_hook))

        if need_attn_out:
            def attn_out_hook(module, inp, out, layer_idx=layer_idx):
                attn_out_layer = out[0] if isinstance(out, tuple) else out
                attn_out[layer_idx] = attn_out_layer[0].detach()
            handles.append(blk.self_attn.register_forward_hook(attn_out_hook))

        if need_head_out:
            def oproj_pre_hook(module, inp, layer_idx=layer_idx):
                flat_heads[layer_idx] = inp[0][0].detach()
            handles.append(blk.self_attn.o_proj.register_forward_pre_hook(oproj_pre_hook))

        if need_mlp_out:
            def mlp_out_hook(module, inp, out, layer_idx=layer_idx):
                y = out[0] if isinstance(out, tuple) else out
                mlp_out[layer_idx] = y[0].detach()
            handles.append(blk.mlp.register_forward_hook(mlp_out_hook))

        if need_resid_mid:
            def resid_mid_hook(module, inp, layer_idx=layer_idx):
                resid_mid[layer_idx] = inp[0][0].detach()
            handles.append(blk.post_attention_layernorm.register_forward_pre_hook(resid_mid_hook))

        if need_resid_post:
            def resid_post_hook(module, inp, out, layer_idx=layer_idx):
                y = out[0] if isinstance(out, tuple) else out
                resid_post[layer_idx] = y[0].detach()
            handles.append(blk.register_forward_hook(resid_post_hook))

            if layer_idx == 0:
                def resid_pre_hook(module, inp, layer_idx=layer_idx):
                    resid_pre0[0] = inp[0][0].detach()
                handles.append(blk.input_layernorm.register_forward_pre_hook(resid_pre_hook))

    need_output_attentions = need_attn_map or (do_knockout and attention_knockout_stage == "post_softmax")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=False,
            output_attentions=need_output_attentions,
        )

    for h in handles:
        h.remove()
    for h in knockout_handles:
        h.remove()
    if embed_handle is not None:
        embed_handle.remove()

    if patched_attn is not None and patched_forward_orig is not None:
        patched_attn.forward = patched_forward_orig

    to_cpu = lambda x: x.detach().cpu()

    # attn_map
    attn_map_t = None
    if need_attn_map:
        attn_map_t = torch.stack([a[0] for a in outputs.attentions])  # [L,H,S,S]
        if "attn_map" in requested:
            cache["attn_map"] = to_cpu(attn_map_t)

    # resid_post
    resid_post_t = None
    if need_resid_post:
        resid_post_t = torch.stack([resid_pre0[0]] + resid_post)  # [L+1,S,d]
        if "resid_post" in requested:
            cache["resid_post"] = to_cpu(resid_post_t)

    # resid_mid
    resid_mid_t = None
    if need_resid_mid:
        resid_mid_t = torch.stack(resid_mid)  # [L,S,d]
        if "resid_mid" in requested:
            cache["resid_mid"] = to_cpu(resid_mid_t)

    # attn_out
    if need_attn_out:
        attn_out_t = torch.stack(attn_out)
        if "attn_out" in requested:
            cache["attn_out"] = to_cpu(attn_out_t)

    # mlp_out
    if need_mlp_out:
        mlp_out_t = torch.stack(mlp_out)
        if "mlp_out" in requested:
            cache["mlp_out"] = to_cpu(mlp_out_t)

    # vproj_out
    vproj_out_t = None
    if need_vproj_out:
        # normalize each captured tensor from [B,S,D] -> [S,D]
        vproj_out_norm = []
        for l in range(L):
            v = vproj_out[l]
            if v is None:
                raise RuntimeError(f"vproj_out[{l}] is None; v_proj hook did not fire.")
            if v.ndim == 3:
                if v.shape[0] != 1:
                    raise RuntimeError(f"Expected batch dim 1 for vproj_out[{l}], got {tuple(v.shape)}")
                v = v[0]
            elif v.ndim != 2:
                raise RuntimeError(f"Unexpected vproj_out[{l}] shape: {tuple(v.shape)}")
            vproj_out_norm.append(v)
        vproj_out_t = torch.stack(vproj_out_norm)  # [L,S,D]
        if "vproj_out" in requested:
            cache["vproj_out"] = to_cpu(vproj_out_t)

    # attn_x_value
    if "attn_x_value" in requested:
        if attn_map_t is None or vproj_out_t is None:
            raise RuntimeError("attn_x_value requested but attn_map/vproj_out not available.")

        attn_x_value_layers = []
        for l in range(L):
            A = attn_map_t[l]  # [H,q,kv]
            Vp = vproj_out_t[l]  # [S,D]

            H = A.shape[0]
            kv_len = A.shape[2]

            Vp_k = Vp[:kv_len]
            D_v = Vp_k.shape[1]  # H_kv * hd (== H * hd only for plain MHA)

            W = llm.layers[l].self_attn.o_proj.weight  # [D_model, H*hd]
            hd = W.shape[1] // H
            if D_v % hd != 0:
                raise RuntimeError(f"Layer {l}: D_v={D_v} not divisible by head_dim={hd}")
            H_kv = D_v // hd

            V = Vp_k.view(kv_len, H_kv, hd)
            if H_kv != H:
                # Grouped-query attention (mistral, llama3): value heads are
                # shared across query-head groups; expand to per-query-head
                # (same as the model's internal repeat_kv).
                V = V.repeat_interleave(H // H_kv, dim=1)
            V = V.permute(1, 0, 2).contiguous()  # [H,kv,hd]

            W_grouped = W.view(W.shape[0], H, hd).permute(1, 0, 2).contiguous()  # [H,D_model,hd]

            # Chunk over heads: the full [H,kv,D] fp32 intermediate (~0.5 GB/layer
            # for 40-head llava-1.5-13b) OOMs 40 GB GPUs; per-chunk math is identical.
            VW_norm = torch.empty(H, kv_len, dtype=A.dtype, device=V.device)
            for h0 in range(0, H, 4):
                VW_chunk = torch.einsum(
                    "hkd,hDd->hkD", V[h0:h0 + 4].float(), W_grouped[h0:h0 + 4].float()
                )  # [<=4,kv,D]
                VW_norm[h0:h0 + 4] = torch.linalg.vector_norm(
                    VW_chunk, ord=2, dim=-1
                ).to(dtype=A.dtype)
                del VW_chunk

            AXV = A * VW_norm[:, None, :]
            attn_x_value_layers.append(AXV)

        cache["attn_x_value"] = to_cpu(torch.stack(attn_x_value_layers, dim=0))

    # head_out
    if "head_out" in requested:
        if flat_heads is None:
            raise RuntimeError("head_out requested but flat_heads not captured.")
        head_out_layers = []
        for layer_idx, blk in enumerate(llm.layers):
            flat = flat_heads[layer_idx]
            if flat is None:
                raise RuntimeError(f"flat_heads[{layer_idx}] is None; o_proj pre-hook did not fire.")
            H = int(outputs.attentions[layer_idx][0].shape[0])
            S_here, D = flat.shape
            if D % H != 0:
                raise RuntimeError(f"D={D} not divisible by H={H} at layer {layer_idx}")
            hd = D // H

            heads = flat.view(S_here, H, hd)
            W = blk.self_attn.o_proj.weight
            W_grouped = W.view(D, H, hd).permute(1, 0, 2).contiguous()
            contrib = torch.einsum("shd,hcd->shc", heads, W_grouped)
            # Offload per layer: accumulating [S,H,D] contributions on GPU and
            # stacking there (~10 GB + copy for 40-layer models) OOMs 40 GB GPUs.
            head_out_layers.append(contrib.permute(1, 0, 2).detach().cpu())
            del contrib

        cache["head_out"] = torch.stack(head_out_layers)

    # embed
    if "embed" in requested:
        if embed[0] is None:
            raise RuntimeError("embed requested but embedding hook did not fire.")
        cache["embed"] = to_cpu(embed[0])

    # interleaved resid
    if "resid" in requested:
        if resid_post_t is None or resid_mid_t is None:
            raise RuntimeError("resid requested but resid_post/resid_mid not available.")
        resid_interleaved = []
        for i in range(len(resid_post_t)):
            resid_interleaved.append(resid_post_t[i])
            if i < len(resid_mid_t):
                resid_interleaved.append(resid_mid_t[i])
        cache["resid"] = to_cpu(torch.stack(resid_interleaved))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for key, value in cache.items():
            torch.save(value, os.path.join(save_dir, f"{key}.pt"))

    return cache


def show_ablation_of_image_tokens(
    tokenizer,
    # prompt: str,
    image: Image.Image,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    ):
    
    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid"], f"Invalid cache key: {cache_key}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")
    
    # # Tokenize the prompt

    # Find the image token and replace it with image tokens
    img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches


    logitdiff_image = ablated_store[:, 5:5+img_token_count, :] - original_store[None, None, :]
    for i, token in enumerate(tokens_of_interest.keys()):
        # logitlens for image tokens
        fig, ax = plt.subplots(6, 6, figsize=(10, 12))

        # flatten ax
        ax = ax.flatten()

        abs_max = np.max(np.abs(logitdiff_image[:, :, i]))
        for l in range(len(ax)):
            if l >= logitdiff_image.shape[0]:
                ax[l].axis("off")
            else:
                ax[l].imshow(np.asarray(image.convert("RGB").resize((image_size, image_size))))
                logitdiff_map = logitdiff_image[l, :, i].reshape(image_size//patch_size, image_size//patch_size)
                logitdiff_map_up = np.kron(logitdiff_map, np.ones((patch_size, patch_size)))
                last_im = ax[l].imshow(logitdiff_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                ax[l].set_xticks([])
                ax[l].set_yticks([])
                if cache_key == "resid_post":
                    if l == 0:
                        ax[l].set_title("resid_pre")
                    else:
                        ax[l].set_title(f"L{l-1}")
                else:
                    ax[l].set_title(f"L{l}")

        cbar = fig.colorbar(
            last_im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"logit diff of {token}"
        )
        if title is not None:
            fig.suptitle(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_image = logitdiff_image[:, :, t0_idx] - logitdiff_image[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_image))

            # flatten ax
            fig, ax = plt.subplots(6, 6, figsize=(10, 12))
            ax = ax.flatten()

            for l in range(len(ax)):
                if l >= two_token_logitdiff_image.shape[0]:
                    ax[l].axis("off")
                else:
                    ax[l].imshow(np.asarray(image.convert("RGB").resize((image_size, image_size))))
                    logitdiff_map = two_token_logitdiff_image[l].reshape(image_size//patch_size, image_size//patch_size)
                    logitdiff_map_up = np.kron(logitdiff_map, np.ones((patch_size, patch_size)))
                    last_im = ax[l].imshow(logitdiff_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                    ax[l].set_xticks([])
                    ax[l].set_yticks([])
                    if cache_key == "resid_post":
                        if l == 0:
                            ax[l].set_title("resid_pre")
                        else:
                            ax[l].set_title(f"L{l-1}")
                    else:
                        ax[l].set_title(f"L{l}")

            cbar = fig.colorbar(
                last_im, ax=ax, location="right",
                fraction=0.03,
                pad=0.02,
                aspect=30,
                shrink=0.9,
                label=f"logit difference ({t0_token} - {t1_token})"
            )
            if title is not None:
                fig.suptitle(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()

def show_ablation_of_text_tokens(
    tokenizer,
    prompt: str,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    text_prompt_length: int = None,
    text_token_labels: list = None,
    ):
    # text_prompt_length/text_token_labels: pass the REAL post-image text-token
    # count and labels (from the processor-expanded input_ids); the fallback
    # below assumes 576 image tokens and img_token_id=32000, which is wrong for
    # anyres models (llava-v1.6/llava-next).

    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid", "head_out"], f"Invalid cache key: {cache_key}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")

    if text_prompt_length is not None:
        assert text_token_labels is not None and len(text_token_labels) == text_prompt_length
        text_prompt_token_list = list(text_token_labels)
    else:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt)

        # Find the image token and replace it with image tokens
        img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches

        token_labels = []
        for token_id in input_ids:
            if token_id == img_token_id:
                # One indexed because the HTML logic wants it that way
                token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
            else:
                token_labels.append(tokenizer.decode([token_id]))

        text_prompt_length = len(token_labels)-(5+img_token_count) # len after <IMG>
        text_prompt_token_list = [token_labels[i] for i in range((len(token_labels)-text_prompt_length), len(token_labels))]


    logitdiff_text = ablated_store[:, -text_prompt_length:, :] - original_store[None, None, :] # (L, H, T)

    L = logitdiff_text.shape[0]

    for i, token in enumerate(tokens_of_interest.keys()):
        plt.figure(figsize=(10, 10))
        abs_max = np.max(np.abs(logitdiff_text[:, :, i]))
        plt.imshow(logitdiff_text[:, :, i].T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        plt.xlabel("layer")
        plt.ylabel("token")
        if cache_key == "resid_post":
            ticks = [0] + (np.arange(0, L, 5) + 1).tolist()
            labels = ["pre"] + np.arange(0, L, 5).tolist()
            plt.xticks(ticks, labels, rotation=0)
        plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
        plt.colorbar(label=f"logit diff of {token}", shrink=0.5)
        if title is not None:
            plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_text = logitdiff_text[:, :, t0_idx] - logitdiff_text[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_text))

            plt.figure(figsize=(10, 10))
            plt.imshow(two_token_logitdiff_text.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
            plt.xlabel("layer")
            plt.ylabel("token")
            if cache_key == "resid_post":
                ticks = [0] + (np.arange(0, L, 5) + 1).tolist()
                labels = ["pre"] + np.arange(0, L, 5).tolist()
                plt.xticks(ticks, labels, rotation=0)
            plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
            plt.colorbar(label=f"logit diff ({t0_token} - {t1_token})", shrink=0.5)
            if title is not None:
                plt.title(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()

def show_ablation_of_heads(
    tokenizer,
    prompt: str,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    # cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    # image_size: int = 336,
    # patch_size: int = 14,
    # img_token_id: int = 32000,
    ):
    
    # # Tokenize the prompt

    # # Find the image token and replace it with image tokens

    logitdiff_heads = ablated_store[:, :, :] - original_store[None, None, :] # (L, H, T)

    for i, token in enumerate(tokens_of_interest.keys()):
        plt.figure(figsize=(10, 10))
        abs_max = np.max(np.abs(logitdiff_heads[:, :, i]))
        plt.imshow(logitdiff_heads[:, :, i].T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        plt.xlabel("layer")
        plt.ylabel("head")
        plt.colorbar(label=f"logit diff of {token}", shrink=0.5)
        if title is not None:
            plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_heads = logitdiff_heads[:, :, t0_idx] - logitdiff_heads[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_heads))

            plt.figure(figsize=(10, 10))
            plt.imshow(two_token_logitdiff_heads.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
            plt.xlabel("layer")
            plt.ylabel("head")
            plt.colorbar(label=f"logit diff ({t0_token} - {t1_token})", shrink=0.5)
            if title is not None:
                plt.title(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()
# resampling ablation

# attention knockout
def collect_knockout_head_out_dla_by_layer(
    model,
    processor,
    image,
    prompt,
    device="cpu",
    attention_knockout_key: Literal["image", "query", "gen_but_last", "last"] = "image",
    attention_knockout_query: Literal["last", "all"] = "all",
    attention_knockout_mode: Literal["drop", "keep_only"] = "keep_only",
    attention_knockout_stage: Literal["pre_softmax", "post_softmax"] = "post_softmax",
    img_token_id: Optional[int] = 32000,
    tokens_of_interest: dict[str, int] = None,
    pos: list[int] = None,
    save_path: str = None,
) -> torch.Tensor:
    """
    Returns: head_out_knockout[L,H,S,d], where entry [l] is head_out at layer l under knockout-at-layer-l.
    """
    if save_path is not None and os.path.exists(save_path):
        print(f"Loading head_out from {save_path}")
        return np.load(save_path)
    if save_path is None:
        print("save_path: None (not using saved file)")
    else:
        print(f"save_path: {save_path}, os.path.exists(save_path): {os.path.exists(save_path)}")

    llm = model.language_model.model
    L = len(llm.layers)

    head_out_dla = []

    # Only resid_post is consumed below; requesting all keys (the default)
    # built ~11 GB of unused per-key stacks per call and OOMed 40-layer models.
    original_cache = get_cache(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        device=device,
        save_dir=None,
        cache_keys=["resid_post"],
    )

    for l in range(L):
        # Only head_out is consumed from the per-layer knockout cache.
        cache_l = get_cache(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            save_dir=None,
            cache_keys=["head_out"],
            attention_knockout_layer=l,
            attention_knockout_key=attention_knockout_key,
            attention_knockout_query=attention_knockout_query,
            attention_knockout_mode=attention_knockout_mode,
            attention_knockout_stage=attention_knockout_stage,
            img_token_id=img_token_id,
        )
        # cache["head_out"] is [L,H,S,d] on CPU; take only the knocked-out layer
        head_out_l = cache_l["head_out"][l:l+1]
        # logitlens returns (1,P,H,d) -> here (1,P,H,T)
        head_out_dla_l = logitlens(
            model=model,
            hidden_states=head_out_l,
            tokens_of_interest=tokens_of_interest,
            pos=pos,
            softmax=False,
            head_out=True,
            device=device,
            ln_mode="cached",
            ln_ref=original_cache["resid_post"][-1],
        )  # numpy (1,H,P,T)
        head_out_dla.append(head_out_dla_l[0]) # (H,P,T)
    head_out_dla = np.stack(head_out_dla, axis=0)  # [L,H,P,T]
    if save_path is not None:
        np.save(save_path, head_out_dla)

    return head_out_dla 

