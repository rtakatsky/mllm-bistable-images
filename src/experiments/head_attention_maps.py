import glob
import os
from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


from utils import *
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List, Dict, Optional, Union, Literal
import random
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib as mpl
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


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
    print(f"save_path: {save_path}, os.path.exists(save_path): {os.path.exists(save_path)}")
    llm = model.language_model.model
    L = len(llm.layers)

    head_out_dla = []

    original_cache = get_cache(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        device=device,
        save_dir=None,
    )

    for l in range(L):
        cache_l = get_cache(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            save_dir=None,
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


def plot_layer_trajectory(
    logits_of_interest: np.ndarray, # resid
    tokens_of_interest: dict[str, int],
    x_token: str,
    y_token: str,
    original_logits: np.ndarray = None,
    cmap: str = "viridis",
    annotate_layers: bool = True,
    image: Image.Image = None,
    save_path: str = None,
    title: str = None,
    softmax: bool = False,
):
    """
    logits_of_interest: shape (num_layers, num_tokens)
    tokens: list of token strings in the same order as columns of logits_of_interest
    x_token, y_token: which two tokens to use for the x/y axes
    """
    if x_token not in tokens_of_interest or y_token not in tokens_of_interest:
        raise ValueError("Requested tokens must be in the provided tokens list.")

    ix = list(tokens_of_interest.keys()).index(x_token)
    iy = list(tokens_of_interest.keys()).index(y_token)
    x = logits_of_interest[:, ix]
    y = logits_of_interest[:, iy]
    layers = np.arange(len(x))  # 0..num_layers-1

    # Build segments for gradient line
    points = np.stack([x, y], axis=1)  # (L, 2)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # (L-1, 2, 2)

    norm = Normalize(vmin=0, vmax=len(layers) - 1)
    lc = LineCollection(segments, colors='red', norm=norm)
    lc.set_array(layers[:-1])  # color between layer i and i+1
    lc.set_linewidth(1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)

    # Scatter the actual layer points with same colormap
    sc = ax.scatter(x, y, c=layers, cmap=cmap, norm=norm, s=20, edgecolor="k", linewidth=0.5)

    # Final layer point
    final_layer = layers[-1]
    final_color = plt.cm.get_cmap(cmap)(norm(final_layer))
    ax.scatter(
        [x[-1]], [y[-1]],
        s=100, marker="X",
        facecolors="none", edgecolors=["red"],
        linewidths=1.0, zorder=10
    )

    if original_logits is not None:
        original_x = original_logits[:, ix]
        original_y = original_logits[:, iy]

        # Build original segments
        orig_points = np.stack([original_x, original_y], axis=1)          # (L, 2)
        orig_segments = np.stack([orig_points[:-1], orig_points[1:]], axis=1)  # (L-1, 2, 2)

        # Same colormap + norm, but different linestyle
        lc_orig = LineCollection(orig_segments, colors='blue', norm=norm)
        lc_orig.set_array(layers[:-1])
        lc_orig.set_linewidth(1)
        lc_orig.set_alpha(0.9)
        ax.add_collection(lc_orig)

        # Markers: scatter doesn't support linestyle, so distinguish by marker/facecolor/alpha
        ax.scatter(
            original_x, original_y,
            c=layers, cmap=cmap, norm=norm,
            s=20, edgecolor="k", linewidth=0.5, alpha=0.9
        )

        # Final layer point
        ax.scatter(
            [original_x[-1]], [original_y[-1]],
            s=60, marker="*",
            facecolors="none", edgecolors=["black"],
            linewidths=1.0, zorder=10
        )

    if annotate_layers:
        for layer_idx in layers:
            if layer_idx == 0:
                annotation = "pre"
            elif layer_idx % 2 == 0:
                annotation = f"{layer_idx//2 - 1}p"
            else:
                annotation = f"{layer_idx//2}m"
            ax.annotate(
                annotation,
                (x[layer_idx], y[layer_idx]),
                fontsize=6,
                xytext=(2, 2),
                textcoords="offset points",
            )
    
    lims = [
        min(np.nanmin(x), np.nanmin(y)),
        max(np.nanmax(x), np.nanmax(y)),
    ]
    ax.plot(lims, lims, linestyle='--', linewidth=1, color='gray')

    ax.set_xlabel(f"logit of {x_token}") if not softmax else ax.set_xlabel(f"prob of {x_token}")
    ax.set_ylabel(f"logit of {y_token}") if not softmax else ax.set_ylabel(f"prob of {y_token}")
    if title is not None:
        ax.set_title(title, loc="left")
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(sc, ax=ax, label="layer", shrink=0.5)

    if image is not None:
        # Inset image
        image_bbox = [0.39, 0.95, 0.2, 0.2]
        inset_ax = fig.add_axes(image_bbox, anchor='NE')
        inset_ax.imshow(image)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    
    plt.close()


def show_logitlens_of_heads(
    logits_heads: np.ndarray,                 # (L, H, T)
    tokens_of_interest: dict[str, int],
    show_diff: list[list[int]] = None,        # e.g. [[0,1]]
    save_dir: str = None,
    title: str = None,
    softmax: bool = False,
):
    """
    Visualize head-wise logit lens as heatmaps over (layer x head) for each token.

    logits_heads: (L, H, T)
      - L: number of layers
      - H: number of heads
      - T: number of tokens_of_interest (in the same key order)
    """
    assert logits_heads.ndim == 3, f"logits_heads must be (L,H,T), got {logits_heads.shape}"
    L, H, T = logits_heads.shape
    assert T == len(tokens_of_interest), (
        f"T mismatch: logits_heads has T={T}, tokens_of_interest has {len(tokens_of_interest)}"
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    token_list = list(tokens_of_interest.keys())

    # Per-token heatmaps
    for i, token in enumerate(token_list):
        mat = logits_heads[:, :, i].T  # (H, L)

        plt.figure(figsize=(10, 6))
        # Use diverging only if you expect signed values; logits are signed, probs aren't.
        if softmax:
            plt.imshow(mat, aspect="auto")
        else:
            abs_max = np.nanmax(np.abs(mat))
            plt.imshow(mat, cmap="RdBu", vmin=-abs_max, vmax=abs_max, aspect="auto")

        plt.xlabel("layer")
        plt.ylabel("head")
        plt.colorbar(
            label=(f"prob of {token}" if softmax else f"logit of {token}"),
            shrink=0.7
        )
        if title is not None:
            plt.title(f"{title}\n{token}")

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"heads_{token}.png"),
                        bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    # Token-pair diffs (e.g. duck - rabbit)
    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0 = token_list[t0_idx]
            t1 = token_list[t1_idx]

            mat = (logits_heads[:, :, t0_idx] - logits_heads[:, :, t1_idx]).T  # (H, L)
            abs_max = np.nanmax(np.abs(mat))

            plt.figure(figsize=(10, 6))
            plt.imshow(mat, cmap="RdBu", vmin=-abs_max, vmax=abs_max, aspect="auto")
            plt.xlabel("layer")
            plt.ylabel("head")
            plt.colorbar(
                label=(f"prob ({t0} - {t1})" if softmax else f"logit ({t0} - {t1})"),
                shrink=0.7
            )
            if title is not None:
                plt.title(f"{title}\n({t0} - {t1})")

            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"heads_{t0}_{t1}.png"),
                            bbox_inches="tight", pad_inches=0.05)
            else:
                plt.show()
            plt.close()

def show_weighted_image_attention_by_dla(
    tokenizer,
    prompt: str,
    image: Image.Image,
    attn_map: torch.Tensor,               # exact mode: cache["attn_map"]; fallback mode: cache["attn_x_value"]
    logits_heads: np.ndarray,             # (L,H,T) or (L,H,P,T) -- still used for fallback / compatibility
    tokens_of_interest: Dict[str, int],
    show_diff: Optional[List[List[int]]] = ((0, 1),),
    layers: Optional[List[int]] = None,
    query_pos: int = -1,
    top_k: int = 6,
    save_dir: Optional[str] = None,
    title: Optional[str] = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    alpha: float = 0.8,
    prefix: str = "imgattn",
    every_k_layers: int = 5,
    pre_img_tokens: int = 5,
    make_head_panels: bool = True,
    vmax_sum: Optional[float] = None,
    vmax_heads: Optional[float] = None,

    # NEW: exact per-key mode inputs
    model=None,
    vproj_out: Optional[torch.Tensor] = None,      # [L,S,D]
    resid_post_ref: Optional[torch.Tensor] = None, # [S,D] or [D], use cache["resid_post"][-1]
):
    """
    Exact mode (recommended):
      For each layer/head/image-key:
        contrib(key) = attn_map[l,h,q,key] * DLA(OV_{l,h,key})
      where OV_{l,h,key} is the per-key value contribution after W_O.
      Heatmaps are sums of these signed per-key contributions.

    Fallback mode:
      If model/vproj_out/resid_post_ref are not given, uses the old heuristic:
        head-level DLA redistributed over image keys using attn_x_value proportions.

    Notes:
      - Exact mode uses ONLY image-key contributions. Therefore, the sum over image patches
        generally does NOT equal the full head DLA if the head also attends to text keys.
      - Query is the last token by default (query_pos=-1).
    """
    assert attn_map.ndim == 4, f"attn_map must be (L,H,S,S), got {attn_map.shape}"
    L, H, S, S2 = attn_map.shape
    assert S == S2, "attn_map last dims must match (S,S)"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if layers is None:
        layers = list(range(L))
    layers = sorted({(l + L) if l < 0 else l for l in layers})
    assert all(0 <= l < L for l in layers), f"layers out of range for L={L}: {layers}"

    grid = image_size // patch_size
    n_img = grid * grid
    img_start = pre_img_tokens
    img_pos_used = torch.arange(img_start, img_start + n_img, dtype=torch.long)

    base = np.asarray(image.convert("L").resize((image_size, image_size)))

    # whether to use exact per-key mode
    use_exact = (model is not None) and (vproj_out is not None) and (resid_post_ref is not None)

    # ----- helpers -----
    def overlay(ax, patch_map, signed=True, vmin=None, vmax=None, return_im=False):
        ax.imshow(base, cmap="gray")
        up = np.kron(patch_map, np.ones((patch_size, patch_size), dtype=np.float32))
        if signed:
            if vmin is None or vmax is None:
                vm = float(np.max(np.abs(up)) + 1e-9)
                vmin_, vmax_ = -vm, vm
            else:
                vmin_, vmax_ = float(vmin), float(vmax)
            im = ax.imshow(up, cmap="RdBu", vmin=vmin_, vmax=vmax_, alpha=alpha)
        else:
            vmin_ = 0.0 if vmin is None else float(vmin)
            vmax_ = float(np.max(up) + 1e-9) if vmax is None else float(vmax)
            im = ax.imshow(up, cmap="viridis", vmin=vmin_, vmax=vmax_, alpha=alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        if return_im:
            return im
        return None

    def get_w(ti: int) -> np.ndarray:
        """
        Original head-level DLA weights (used for fallback and optional compatibility).
        """
        if logits_heads.ndim == 4:
            P = logits_heads.shape[2]
            qp = (query_pos + P) if query_pos < 0 else query_pos
            assert 0 <= qp < P, f"query_pos {query_pos} out of range for P={P}"
            w = logits_heads[:, :, qp, ti]  # (L,H)
        else:
            w = logits_heads[:, :, ti]      # (L,H)
        assert w.shape[:2] == (L, H), f"logits_heads mismatch: {w.shape} vs {(L,H)}"
        return w.astype(np.float32)

    # ----- exact-mode helpers -----
    if use_exact:
        llm = model.language_model.model
        norm = llm.norm
        lm_head = model.language_model.lm_head
        work_device = next(model.parameters()).device

        token_names = list(tokens_of_interest.keys())
        token_ids = list(tokens_of_interest.values())

        # final reference should be the FINAL pre-final-norm residual at the query position
        # resid_post_ref is expected to be cache["resid_post"][-1] with shape [S,D]
        if resid_post_ref.ndim == 2:
            q_idx_ref = (query_pos + resid_post_ref.shape[0]) if query_pos < 0 else query_pos
            ref_vec = resid_post_ref[q_idx_ref].to(work_device).float()  # [D]
        elif resid_post_ref.ndim == 1:
            ref_vec = resid_post_ref.to(work_device).float()
        else:
            raise ValueError(f"resid_post_ref must be [S,D] or [D], got {tuple(resid_post_ref.shape)}")

        def apply_cached_final_norm_rms(x: torch.Tensor) -> torch.Tensor:
            """
            Fixed-stat RMSNorm contribution map:
              y = (x / rms(ref)) * weight
            This is the correct linearized form for LLaMA/Mistral-style RMSNorm under cached stats.
            """
            eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
            rms = torch.rsqrt(ref_vec.pow(2).mean(dim=-1, keepdim=False) + eps)  # scalar
            y = x * rms
            if hasattr(norm, "weight") and norm.weight is not None:
                y = y * norm.weight.to(y.device, dtype=y.dtype)
            return y

        def build_readout_vector_for_token(ti: int) -> torch.Tensor:
            """
            Return the lm_head readout vector for one token id, no bias.
            """
            tid = token_ids[ti]
            # lm_head.weight: [V,D]
            return lm_head.weight[tid].to(work_device).float()

        def build_readout_vector_for_diff(a: int, b: int) -> torch.Tensor:
            """
            Return lm_head.weight[token_a] - lm_head.weight[token_b]
            """
            return build_readout_vector_for_token(a) - build_readout_vector_for_token(b)

        def compute_exact_head_maps(readout_vec: torch.Tensor):
            """
            Returns:
              head_maps: list of tuples (layer, head, scalar_sum, patch_map)
                scalar_sum = sum over image keys of exact image contribution for that (layer,head)
                patch_map  = [grid,grid]
            """
            head_maps = []

            for l in layers:
                A = attn_map[l]  # [H,S,S], CPU tensor
                q_idx = (query_pos + A.shape[1]) if query_pos < 0 else query_pos
                assert 0 <= q_idx < A.shape[1], f"query_pos {query_pos} out of range for attn_map layer {l}"

                # exact post-softmax attention to image keys for the chosen query
                A_img = A[:, q_idx, :].index_select(1, img_pos_used).float()  # [H,Nimg]

                # reconstruct per-head V for image keys from cached v_proj output
                Vp_l = vproj_out[l]  # [S,D]
                if Vp_l.ndim != 2:
                    raise ValueError(f"vproj_out[{l}] must be [S,D], got {tuple(Vp_l.shape)}")
                Vp_img = Vp_l.index_select(0, img_pos_used).to(work_device).float()  # [Nimg,D]

                D = Vp_img.shape[1]
                if D % H != 0:
                    raise RuntimeError(f"Layer {l}: D={D} not divisible by H={H}")
                hd = D // H

                # [Nimg,H,hd] -> [H,Nimg,hd]
                V_heads = Vp_img.view(n_img, H, hd).permute(1, 0, 2).contiguous()

                # grouped W_O: [H,D,hd]
                W = llm.layers[l].self_attn.o_proj.weight.to(work_device).float()  # [D,D]
                W_grouped = W.view(D, H, hd).permute(1, 0, 2).contiguous()          # [H,D,hd]

                for h in range(H):
                    # per-key OV vectors for this head: [Nimg,D]
                    # OV_{h,k} = W_O^{(h)} v_{h,k}
                    OV = V_heads[h] @ W_grouped[h].transpose(0, 1)  # [Nimg,D]

                    # cached final norm linearization + token readout
                    OV_norm = apply_cached_final_norm_rms(OV)        # [Nimg,D]
                    dla_per_key = OV_norm @ readout_vec              # [Nimg]

                    # exact contribution at the chosen query
                    contrib = dla_per_key * A_img[h].to(work_device) # [Nimg]
                    contrib_np = contrib.detach().cpu().numpy().astype(np.float32)

                    head_maps.append((
                        l,
                        h,
                        float(contrib_np.sum()),
                        contrib_np.reshape(grid, grid),
                    ))

            return head_maps

    # ----- old heuristic fallback -----
    def compute_block_heatmaps_fallback(w_lh: np.ndarray) -> List[np.ndarray]:
        """
        Old behavior:
          redistribute head-level DLA over image keys using attn_x_value proportions.
        Here, `attn_map` is expected to actually be cache["attn_x_value"].
        """
        w_sel = w_lh[layers, :]  # [Lsel,H]

        AXV = attn_map.index_select(0, torch.tensor(layers, dtype=torch.long))  # [Lsel,H,S,S]
        q_idx = (query_pos + AXV.shape[2]) if query_pos < 0 else query_pos
        AXV = AXV[:, :, q_idx, :]                                # [Lsel,H,S]
        AXV_img = AXV.index_select(-1, img_pos_used)             # [Lsel,H,Nimg]
        axv_np = AXV_img.detach().cpu().numpy().astype(np.float32)

        denom = axv_np.sum(axis=-1, keepdims=True)               # [Lsel,H,1]
        denom_safe = np.where(denom > 1e-12, denom, 1.0)
        dist = axv_np / denom_safe                               # [Lsel,H,Nimg]
        contrib = dist * w_sel[:, :, None]                       # [Lsel,H,Nimg]

        blocks = []
        Lsel = len(layers)
        for b0 in range(0, Lsel, every_k_layers):
            b1 = min(b0 + every_k_layers, Lsel)
            block_sum = contrib[b0:b1].sum(axis=(0, 1))          # [Nimg]
            blocks.append(block_sum.reshape(grid, grid))
        return blocks

    # ----- drawing logic -----
    def draw_one(name: str, mode_payload, out_path: Optional[str], exact_mode: bool):
        """
        exact_mode:
          mode_payload = readout_vec (torch.Tensor [D])

        fallback:
          mode_payload = w_lh (np.ndarray [L,H])
        """
        if exact_mode:
            head_maps = compute_exact_head_maps(mode_payload)  # list of (l,h,sum,map)

            # block summaries
            blocks = []
            for b0 in range(0, len(layers), every_k_layers):
                b1 = min(b0 + every_k_layers, len(layers))
                block_layers = set(layers[b0:b1])

                acc = np.zeros((grid, grid), dtype=np.float32)
                for l, h, s, m in head_maps:
                    if l in block_layers:
                        acc += m
                blocks.append(acc)

            nblocks = len(blocks)

            # shared scale for summary
            if vmax_sum is None:
                global_vmax = float(max(np.max(np.abs(hm)) for hm in blocks) + 1e-9)
            else:
                global_vmax = float(vmax_sum)
            vmin, vmax = -global_vmax, global_vmax

            fig = plt.figure(figsize=(3.4 * min(5, nblocks), 3.4 * int(np.ceil(nblocks / min(5, nblocks)))))
            ncols = min(5, nblocks)
            nrows = int(np.ceil(nblocks / ncols))
            gs = fig.add_gridspec(nrows, ncols)

            mappable = None
            for bi, hm in enumerate(blocks):
                r, c = divmod(bi, ncols)
                ax = fig.add_subplot(gs[r, c])
                im = overlay(ax, hm, signed=True, vmin=vmin, vmax=vmax, return_im=True)
                if mappable is None:
                    mappable = im
                l0 = layers[bi * every_k_layers]
                l1 = layers[min((bi + 1) * every_k_layers - 1, len(layers) - 1)]
                ax.set_title(f"L{l0}–L{l1} (sum)", fontsize=10)

            for bi in range(nblocks, nrows * ncols):
                r, c = divmod(bi, ncols)
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")

            if title is not None:
                fig.suptitle(title, fontsize=12, y=0.995)
            else:
                fig.suptitle(name, fontsize=12, y=0.995)

            plt.tight_layout(rect=[0.0, 0.0, 0.88, 0.98])
            if mappable is not None:
                cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                fig.colorbar(mappable, cax=cax, label="exact per-key DLA (sum over heads/layers)")

            if out_path is not None:
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            else:
                plt.show()
            plt.close(fig)

            # ---- plot-input cache (rule: every paper figure must be
            # replottable from cached data without a GPU). Fig 7 is rendered for
            # the paper from this npz by src/make_paper_panels.py; the panels
            # below stay as the in-experiment diagnostic view.
            if save_dir is not None and head_maps:
                os.makedirs(save_dir, exist_ok=True)
                npz_path = os.path.join(save_dir, f"{prefix}_{name}_heads.npz")
                np.savez_compressed(
                    npz_path,
                    layers=np.array([lm[0] for lm in head_maps], dtype=np.int32),
                    heads=np.array([lm[1] for lm in head_maps], dtype=np.int32),
                    sums=np.array([lm[2] for lm in head_maps], dtype=np.float32),
                    maps=np.stack([lm[3] for lm in head_maps]).astype(np.float32),
                    image=np.asarray(image.convert("L").resize((image_size, image_size))),
                    name=np.array(name),
                    query_pos=np.array(query_pos),
                )
                print(f"[cache] head-panel inputs -> {npz_path}")

            # optional head panels ranked by exact IMAGE contribution
            if make_head_panels:
                head_maps_sorted = sorted(head_maps, key=lambda x: x[2], reverse=True)
                top = head_maps_sorted[:top_k]
                worst = sorted(head_maps, key=lambda x: x[2])[:top_k]

                k = max(1, top_k)
                fig2 = plt.figure(figsize=(3.2 * k, 6.6))
                gs2 = fig2.add_gridspec(2, k)

                if vmax_heads is None:
                    all_maps = [m for (_, _, _, m) in (top + worst)]
                    shared_v = float(max((np.max(np.abs(m)) for m in all_maps), default=0.0) + 1e-9)
                else:
                    shared_v = float(vmax_heads)

                mappable2 = None

                for i in range(k):
                    ax = fig2.add_subplot(gs2[0, i])
                    if i < len(top):
                        l, h, s, m = top[i]
                        im = overlay(ax, m, signed=True, vmin=-shared_v, vmax=shared_v, return_im=True)
                        if mappable2 is None:
                            mappable2 = im
                        ax.set_title(f"top {i+1}: L{l} H{h} sum={s:.3g}", fontsize=9)
                    else:
                        ax.axis("off")

                for i in range(k):
                    ax = fig2.add_subplot(gs2[1, i])
                    if i < len(worst):
                        l, h, s, m = worst[i]
                        im = overlay(ax, m, signed=True, vmin=-shared_v, vmax=shared_v, return_im=True)
                        if mappable2 is None:
                            mappable2 = im
                        ax.set_title(f"bottom {i+1}: L{l} H{h} sum={s:.3g}", fontsize=9)
                    else:
                        ax.axis("off")

                plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
                if mappable2 is not None:
                    cax2 = fig2.add_axes([0.90, 0.15, 0.02, 0.70])
                    fig2.colorbar(mappable2, cax=cax2, label="exact per-head image contribution")

                out2 = os.path.join(save_dir, f"{prefix}_{name}_heads.png") if save_dir else None
                if out2 is not None:
                    plt.savefig(out2, bbox_inches="tight", pad_inches=0.05)
                else:
                    plt.show()
                plt.close(fig2)

        else:
            # old heuristic path
            blocks = compute_block_heatmaps_fallback(mode_payload)
            nblocks = len(blocks)

            if vmax_sum is None:
                global_vmax = float(max(np.max(np.abs(hm)) for hm in blocks) + 1e-9)
            else:
                global_vmax = float(vmax_sum)
            vmin, vmax = -global_vmax, global_vmax

            ncols = min(5, nblocks)
            nrows = int(np.ceil(nblocks / ncols))
            fig = plt.figure(figsize=(3.4 * ncols, 3.4 * nrows))
            gs = fig.add_gridspec(nrows, ncols)

            mappable = None
            for bi, hm in enumerate(blocks):
                r, c = divmod(bi, ncols)
                ax = fig.add_subplot(gs[r, c])
                im = overlay(ax, hm, signed=True, vmin=vmin, vmax=vmax, return_im=True)
                if mappable is None:
                    mappable = im
                l0 = layers[bi * every_k_layers]
                l1 = layers[min((bi + 1) * every_k_layers - 1, len(layers) - 1)]
                ax.set_title(f"L{l0}–L{l1} (sum)", fontsize=10)

            for bi in range(nblocks, nrows * ncols):
                r, c = divmod(bi, ncols)
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")

            if title is not None:
                fig.suptitle(title, fontsize=12, y=0.995)

            plt.tight_layout(rect=[0.0, 0.0, 0.88, 0.98])
            if mappable is not None:
                cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                fig.colorbar(mappable, cax=cax, label="heuristic DLA-weighted (sum)")

            if out_path is not None:
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            else:
                plt.show()
            plt.close(fig)

    # ----- per-token -----
    token_list = list(tokens_of_interest.keys())
    os.makedirs(save_dir, exist_ok=True)
    if use_exact:
        for ti, tok in enumerate(token_list):
            readout_vec = build_readout_vector_for_token(ti)
            out = os.path.join(save_dir, f"{tok}.png") if save_dir else None
            draw_one(tok, readout_vec, out, exact_mode=True)

        if show_diff is not None:
            for a, b in show_diff:
                ta, tb = token_list[a], token_list[b]
                readout_vec = build_readout_vector_for_diff(a, b)
                out = os.path.join(save_dir, f"{ta}_minus_{tb}.png") if save_dir else None
                draw_one(f"{ta} - {tb}", readout_vec, out, exact_mode=True)
    else:
        for ti, tok in enumerate(token_list):
            w = get_w(ti)
            out = os.path.join(save_dir, f"{tok}.png") if save_dir else None
            draw_one(tok, w, out, exact_mode=False)

        if show_diff is not None:
            for a, b in show_diff:
                ta, tb = token_list[a], token_list[b]
                w = get_w(a) - get_w(b)
                out = os.path.join(save_dir, f"{ta}_minus_{tb}.png") if save_dir else None
                draw_one(f"{ta} - {tb}", w, out, exact_mode=False)

## temporarily from utils.py
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

from typing import Optional, Literal, Tuple, Dict, List, Sequence, Union
import numpy as np
import math
import os
import torch
import matplotlib.pyplot as plt


def _compute_image_and_text_positions(
    input_ids_1d: torch.Tensor,
    img_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image_pos: all positions occupied by image tokens
    text_pos: all positions strictly after the image-token block
    """
    assert input_ids_1d.dim() == 1
    device = input_ids_1d.device
    S = int(input_ids_1d.numel())

    image_pos = (input_ids_1d == img_token_id).nonzero(as_tuple=True)[0].to(torch.long)
    if image_pos.numel() == 0:
        raise ValueError("No image tokens found in input_ids.")

    last_img = int(image_pos.max().item())
    text_pos = torch.arange(last_img + 1, S, device=device, dtype=torch.long)

    return image_pos, text_pos


def _resolve_positions(
    pos: Union[int, Sequence[int]],
    S: int,
    device,
) -> torch.Tensor:
    if isinstance(pos, int):
        pos = [pos]
    out = []
    for p in pos:
        q = p + S if p < 0 else p
        if not (0 <= q < S):
            raise ValueError(f"Position {p} resolved to {q}, out of range for S={S}")
        out.append(q)
    return torch.tensor(out, device=device, dtype=torch.long)


def _select_vocab_slice(
    logits_2d: torch.Tensor,   # [P, V]
    tokens_of_interest: Optional[Dict[str, int]],
) -> torch.Tensor:
    if tokens_of_interest is None:
        return logits_2d
    ids = torch.tensor(list(tokens_of_interest.values()), device=logits_2d.device, dtype=torch.long)
    return logits_2d.index_select(-1, ids)

def _make_centered_layer_windows(L: int, window_size: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Return centers [0..L-1] and clipped centered windows.
    For odd width=9:
      center 0  -> (0, 4)
      center 1  -> (0, 5)
      ...
      center 4  -> (0, 8)
      ...
      center 31 -> (27, 31)
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    left = window_size // 2
    right = window_size - 1 - left

    centers = list(range(L))
    windows = []
    for c in centers:
        l0 = max(0, c - left)
        l1 = min(L - 1, c + right)
        windows.append((l0, l1))
    return centers, windows

def collect_sliding_window_text_to_image_knockout_logits(
    model,
    processor,
    image,
    prompt,
    device="cpu",
    window_size: int = 9,
    target_pos: Union[int, Sequence[int]] = -1,
    tokens_of_interest: Optional[Dict[str, int]] = None,
    attention_knockout_stage: Literal["pre_softmax", "post_softmax"] = "pre_softmax",
    img_token_id: Optional[int] = None,
    knockout_value: Literal["-inf", "finfo_min"] = "-inf",
    verbose: bool = True,
):
    """
    Centered-window layer knockout:
      - centers run from 0..L-1
      - query positions: all text tokens after the image-token block
      - key positions: all image tokens
      - return output logits at target_pos

    Returns:
      {
        "centers": [L],
        "windows": list[(l0,l1)] length L,
        "baseline_logits": [P,T] or [P,V],
        "window_logits": [L,P,T] or [L,P,V],
        "selected_positions": [P],
        "image_pos": [Nimg],
        "text_pos": [Ntxt],
      }
    """
    llm = model.language_model.model
    L = len(llm.layers)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids_1d = inputs["input_ids"][0]
    S = int(input_ids_1d.numel())

    img_token_id_eff = _infer_img_token_id(processor, model, img_token_id)
    image_pos, text_pos = _compute_image_and_text_positions(input_ids_1d, img_token_id_eff)
    pos_idx = _resolve_positions(target_pos, S, device=input_ids_1d.device)

    if verbose:
        print(f"[scan] seq_len={S}, n_image_tokens={image_pos.numel()}, n_text_tokens={text_pos.numel()}")

    if image_pos.numel() == 0:
        raise ValueError("No image tokens found.")
    if text_pos.numel() == 0:
        raise ValueError("No text tokens found after image tokens.")

    # Baseline
    with torch.no_grad():
        baseline_out = model(**inputs, output_attentions=False)
    baseline_logits = baseline_out.logits[0].index_select(0, pos_idx)  # [P,V]
    baseline_logits = _select_vocab_slice(baseline_logits, tokens_of_interest).detach().cpu()

    centers, windows = _make_centered_layer_windows(L, window_size)
    fill = _get_fill_value(dtype=model.dtype, strategy=knockout_value)

    knocked_logits = []

    for center, (start, end) in zip(centers, windows):
        handles = []
        patched = []

        try:
            if attention_knockout_stage == "pre_softmax":
                def _apply_pre_softmax_mask(args, kwargs):
                    """
                    Robust to self_attn being called with positional args or kwargs.
                    """
                    if kwargs is None:
                        kwargs = {}

                    hidden_states = None
                    if len(args) > 0 and torch.is_tensor(args[0]):
                        hidden_states = args[0]
                    else:
                        hidden_states = kwargs.get("hidden_states", None)

                    if hidden_states is None:
                        raise RuntimeError(
                            "Could not find hidden_states in self_attn pre-hook. "
                            f"len(args)={len(args)}, kwargs.keys()={list(kwargs.keys())}"
                        )

                    B, S_here, _ = hidden_states.shape

                    if S_here != S:
                        raise RuntimeError(
                            f"Sequence length mismatch inside self_attn: "
                            f"processor/input S={S}, self_attn S={S_here}"
                        )

                    attn_mask = kwargs.get("attention_mask", None)

                    # Case 1: no mask provided -> build full additive causal mask
                    if attn_mask is None:
                        mask = torch.zeros(
                            (B, 1, S_here, S_here),
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                        causal = torch.triu(
                            torch.ones(S_here, S_here, device=hidden_states.device, dtype=torch.bool),
                            diagonal=1,
                        )
                        mask = mask.masked_fill(causal[None, None, :, :], fill)

                    # Case 2: padding mask [B,S] -> expand to additive [B,1,S,S]
                    elif attn_mask.dim() == 2:
                        mask = _ensure_4d_additive_mask_from_2d(
                            attn_mask_2d=attn_mask,
                            hidden_dtype=hidden_states.dtype,
                            fill_value=fill,
                        )

                    # Case 3: additive / broadcast mask already provided
                    elif attn_mask.dim() == 4:
                        mask = attn_mask.clone()

                    else:
                        raise RuntimeError(f"Unexpected attention_mask dim: {attn_mask.dim()}")

                    # Knock out text(query) -> image(key)
                    if mask.dtype == torch.bool:
                        mask[:, :, text_pos[:, None], image_pos[None, :]] = True
                    else:
                        fv = torch.tensor(fill, device=mask.device, dtype=mask.dtype) if not isinstance(fill, float) else fill
                        mask[:, :, text_pos[:, None], image_pos[None, :]] = fv

                    kwargs["attention_mask"] = mask
                    return kwargs

                for l in range(start, end + 1):
                    blk = llm.layers[l]

                    def _pre_hook(module, args, kwargs, _apply=_apply_pre_softmax_mask):
                        kwargs = _apply(args, kwargs)
                        return (args, kwargs)

                    try:
                        h = blk.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
                        handles.append(h)
                    except TypeError:
                        patched_attn = blk.self_attn
                        forward_orig = patched_attn.forward

                        def _forward_patched(*args, **kwargs):
                            kwargs = _apply_pre_softmax_mask(args, kwargs)
                            return forward_orig(*args, **kwargs)

                        patched_attn.forward = _forward_patched
                        patched.append((patched_attn, forward_orig))

                with torch.no_grad():
                    out = model(**inputs, output_attentions=False)

            elif attention_knockout_stage == "post_softmax":
                # This path does not depend on attention_mask being present.
                # It is slower but very robust.
                vproj_buf = {}

                for l in range(start, end + 1):
                    blk = llm.layers[l]

                    def _vproj_hook(module, inp, out, layer_idx=l):
                        vproj_buf[layer_idx] = out.detach()

                    handles.append(blk.self_attn.v_proj.register_forward_hook(_vproj_hook))

                    def _rewrite_hook(module, inp, out, layer_idx=l):
                        if not isinstance(out, tuple) or len(out) < 2:
                            return out

                        attn_weights = out[1]
                        rest = out[2:] if len(out) > 2 else ()

                        if attn_weights is None:
                            raise RuntimeError(
                                "post_softmax knockout needs attn_weights. "
                                "Use output_attentions=True and an eager/supported attention backend."
                            )
                        if layer_idx not in vproj_buf:
                            raise RuntimeError(f"Missing v_proj output for layer {layer_idx}")

                        v = vproj_buf[layer_idx]  # [B,kv,D]
                        B, H, q_len, kv_len = attn_weights.shape
                        if v.shape[1] != kv_len:
                            raise RuntimeError(
                                f"Layer {layer_idx}: kv_len mismatch "
                                f"(attn={kv_len}, v_proj={v.shape[1]})"
                            )

                        hd = v.shape[-1] // H
                        value_states = v.view(B, kv_len, H, hd).transpose(1, 2).contiguous()  # [B,H,kv,hd]

                        w = attn_weights.clone()
                        w[:, :, text_pos[:, None], image_pos[None, :]] = 0.0

                        out_heads = torch.matmul(w, value_states)  # [B,H,q,hd]
                        out_concat = out_heads.transpose(1, 2).contiguous().view(B, q_len, H * hd)
                        new_attn_output = module.o_proj(out_concat)

                        if len(out) == 2:
                            return (new_attn_output, w)
                        return (new_attn_output, w, *rest)

                    handles.append(blk.self_attn.register_forward_hook(_rewrite_hook))

                with torch.no_grad():
                    out = model(**inputs, output_attentions=True)

            else:
                raise ValueError(f"Unknown attention_knockout_stage: {attention_knockout_stage}")

            logits = out.logits[0].index_select(0, pos_idx)
            logits = _select_vocab_slice(logits, tokens_of_interest).detach().cpu()
            knocked_logits.append(logits)

        finally:
            for h in handles:
                h.remove()
            for patched_attn, forward_orig in patched:
                patched_attn.forward = forward_orig

    return {
        "centers": centers,
        "windows": windows,
        "baseline_logits": baseline_logits.numpy(),
        "window_logits": torch.stack(knocked_logits, dim=0).numpy(),  # [L,P,T]
        "selected_positions": pos_idx.detach().cpu().numpy(),
        "image_pos": image_pos.detach().cpu().numpy(),
        "text_pos": text_pos.detach().cpu().numpy(),
    }

def plot_sliding_window_knockout_effect(
    scan_result: dict,
    tokens_of_interest: Dict[str, int],
    pos_index: int = 0,
    show_diff: Optional[List[Tuple[int, int]]] = None,
    use_delta: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    x-axis = window center (0..L-1), not window span labels.
    """
    centers = np.array(scan_result["centers"])
    windows = scan_result["windows"]
    baseline = scan_result["baseline_logits"][pos_index]   # [T]
    arr = scan_result["window_logits"][:, pos_index, :]    # [L,T]

    token_names = list(tokens_of_interest.keys())

    plt.figure(figsize=(12, 4.8))

    for i, tok in enumerate(token_names):
        y = arr[:, i] - baseline[i] if use_delta else arr[:, i]
        plt.plot(centers, y, marker="o", linewidth=1.5, label=tok)

    if show_diff is not None:
        for a, b in show_diff:
            name = f"{token_names[a]} - {token_names[b]}"
            y = arr[:, a] - arr[:, b]
            if use_delta:
                y = y - (baseline[a] - baseline[b])
            plt.plot(centers, y, marker="o", linewidth=2.2, label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    plt.xlim(centers.min(), centers.max())
    plt.xticks(np.arange(centers.min(), centers.max() + 1, 1))
    plt.xlabel("window center layer")
    plt.ylabel("Δ logit" if use_delta else "logit")

    if title is not None:
        plt.title(title)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


def _token_str_for_title(tokenizer, token_id: int) -> str:
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    if tok is None:
        return str(token_id)
    return tok


def _build_token_context_pieces(
    tokenizer,
    input_ids_1d: torch.Tensor,
    qpos: int,
    context_radius: int = 3,
):
    """
    Returns a list of (text, color) pieces.
    The selected token is red.
    """
    S = int(input_ids_1d.numel())
    left = max(0, qpos - context_radius)
    right = min(S, qpos + context_radius + 1)

    pieces = []
    if left > 0:
        pieces.append(("… ", "black"))

    for i in range(left, right):
        tok = _token_str_for_title(tokenizer, int(input_ids_1d[i].item()))
        color = "red" if i == qpos else "black"
        suffix = " " if i < right - 1 else ""
        pieces.append((tok + suffix, color))

    if right < S:
        pieces.append((" …", "black"))

    return pieces


def _set_context_title(
    ax,
    pieces,
    layer_text: str,
    fontsize_context: int = 9,
    fontsize_layer: int = 10,
    y: float = 1.08,
):
    """
    First line: token context (with one token red)
    Second line: layer text
    """
    line1_children = [
        TextArea(txt, textprops=dict(color=color, fontsize=fontsize_context))
        for txt, color in pieces
    ]
    line1 = HPacker(children=line1_children, align="center", pad=0, sep=0)
    line2 = TextArea(layer_text, textprops=dict(color="black", fontsize=fontsize_layer))

    packed = VPacker(children=[line1, line2], align="center", pad=0, sep=1)

    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=packed,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored)

def show_window_mean_image_attention(
    model,
    processor,
    image,
    prompt,
    windows: List[Tuple[int, int]],
    device="cpu",
    mode: Literal["raw_attention", "contribution"] = "raw_attention",
    extra_query_pos: Optional[int] = None,
    img_token_id: Optional[int] = None,
    patch_size: int = 14,
    alpha: float = 0.8,
    cache_save_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Visualize mean image attention / mean image contribution over specified layer windows.

    mode:
      - "raw_attention": use cache["attn_map"]
      - "contribution": use cache["attn_x_value"] = attention * ||OV(value)||

    Query rows shown by default:
      - last token
      - mean over all text tokens after image tokens
      - optional extra_query_pos (e.g. -7)
    """
    if mode == "raw_attention":
        cache_keys = ["attn_map"]
        src_key = "attn_map"
        cbar_label = "mean attention to image"
    elif mode == "contribution":
        cache_keys = ["attn_x_value"]
        src_key = "attn_x_value"
        cbar_label = r"mean attention × ||OV(value)||"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cache = get_cache(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        device=device,
        save_dir=cache_save_dir,
        cache_keys=cache_keys,
    )
    src = cache[src_key]  # [L,H,S,S], on CPU

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids_1d = inputs["input_ids"][0]
    S = int(input_ids_1d.numel())

    img_token_id_eff = _infer_img_token_id(processor, model, img_token_id)
    image_pos, text_pos = _compute_image_and_text_positions(input_ids_1d, img_token_id_eff)
    image_pos = image_pos.cpu()
    text_pos = text_pos.cpu()

    if text_pos.numel() == 0:
        raise ValueError("No text tokens found after the image-token block.")

    n_img = int(image_pos.numel())
    grid = int(round(math.sqrt(n_img)))
    if grid * grid != n_img:
        raise ValueError(
            f"Number of image tokens ({n_img}) is not a square. "
            "This plotting helper assumes a square image-token grid."
        )


    query_specs = [
        ("last", torch.tensor([S - 1], dtype=torch.long), "simple", S - 1),
        ("mean_text", text_pos, "simple", None),
    ]
    if extra_query_pos is not None:
        q = extra_query_pos + S if extra_query_pos < 0 else extra_query_pos
        if not (0 <= q < S):
            raise ValueError(f"extra_query_pos={extra_query_pos} resolved to {q}, out of range for S={S}")
        query_specs.append(("extra_context", torch.tensor([q], dtype=torch.long), "context", q))

    base = np.asarray(image.convert("L").resize((grid * patch_size, grid * patch_size)))

    maps = {}

    for qname, qpos, _, _ in query_specs:
        maps[qname] = []
        for l0, l1 in windows:
            if not (0 <= l0 <= l1 < src.shape[0]):
                raise ValueError(f"Window {(l0, l1)} out of range for L={src.shape[0]}")

            layer_idx = torch.arange(l0, l1 + 1, dtype=torch.long)
            block = src.index_select(0, layer_idx)                 # [W,H,S,S]
            block = block.index_select(2, qpos)                    # [W,H,Q,S]
            block = block.index_select(3, image_pos)               # [W,H,Q,Nimg]

            patch_map = block.mean(dim=(0, 1, 2)).numpy().astype(np.float32)  # [Nimg]
            patch_map = patch_map.reshape(grid, grid)

            maps[qname].append(patch_map)

    nrows = len(query_specs)
    ncols = len(windows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.3 * ncols, 3.3 * nrows),
        squeeze=False,
    )

    for r, (qname, _, title_mode, q_single) in enumerate(query_specs):
        for c, (l0, l1) in enumerate(windows):
            ax = axes[r, c]
            hm = maps[qname][c]

            ax.imshow(base, cmap="gray")
            up = np.kron(hm, np.ones((patch_size, patch_size), dtype=np.float32))

            local_vmax = float(np.max(up) + 1e-9)
            ax.imshow(up, cmap="viridis", vmin=0.0, vmax=local_vmax, alpha=alpha)

            if title_mode == "simple":
                ax.set_title(f"{qname}\nL{l0}–L{l1}", fontsize=10)
            else:
                pieces = _build_token_context_pieces(
                    tokenizer=processor.tokenizer,
                    input_ids_1d=input_ids_1d.detach().cpu(),
                    qpos=q_single,
                    context_radius=3,
                )
                _set_context_title(
                    ax=ax,
                    pieces=pieces,
                    layer_text=f"L{l0}–L{l1}",
                    fontsize_context=8,
                    fontsize_layer=10,
                    y=1.08,
                )

            ax.set_xticks([])
            ax.set_yticks([])

    if title is not None:
        fig.suptitle(title, y=0.995, fontsize=12)

    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.97])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return maps


from pathlib import Path
import os
import argparse
import torch


def get_base_dir(cli_base_dir: str | None = None) -> Path:
    """
    Priority:
      1. --base_dir
      2. VLM_BISTABLE_BASE_DIR env var
      3. infer from this script location
    Assumes this script is in <base_dir>/src/
    """
    if cli_base_dir:
        return Path(cli_base_dir).expanduser().resolve()

    env_base_dir = os.environ.get("VLM_BISTABLE_BASE_DIR")
    if env_base_dir:
        return Path(env_base_dir).expanduser().resolve()

    # this file: <base_dir>/src/experiments/head_attention_maps.py
    return Path(PROJECT_ROOT)

if __name__ == "__main__":
    from utils import *
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()
    model_id = args.model

    base_dir = get_base_dir(args.base_dir)
    print(f"base_dir: {base_dir}")

    os.environ["HF_HOME"] = str(base_dir / "huggingface_cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bistable_image_dir = base_dir / "data" / "duck_rabbit"
    # This experiment runs on the canonical duck-rabbit only.
    bistable_image_paths = [bistable_image_dir / f"{image_name}.png" for image_name in ["harper"]]

    results_dir = base_dir / "outputs" / "head_attention_maps"
    os.makedirs(results_dir, exist_ok=True)

    # Prompts
    prompts=[
        ["List every animal in the image, each in one word.", "I see a"],
        ["List every animal in the image, each in one word. It starts with 'r'.", "I see a"],
        ["List every animal in the image, each in one word. It starts with 'd'.", "I see a"],
    ]

    vlm_dict={
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        # "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",

        # "llava-v1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
        # "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        # "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf",
    }

    # %%
  # Skip (prompt, model) pairs whose results already exist.
  # Build the (prompt, model) task list.
    tasks = [
        (image_path, text_prompt)
        for image_path in bistable_image_paths
        for text_prompt in prompts
    ]

    for model_id in vlm_dict.keys():
        print(f"Running {model_id}...")
        processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
        model.to(device)
        model.eval()
        tokenizer = processor.tokenizer
        tokens_of_interest = {token: tokenizer.convert_tokens_to_ids(token) for token in ["▁bird", "▁rabb", "▁du", "▁b"]}
        for image_path, text_prompt in tqdm(tasks, desc='All tasks'):
            image_name = os.path.basename(image_path).split('.')[0]
            prompt, prompt_prefix = text_prompt[0], text_prompt[1]
            text_input = f"USER: <image>\n{prompt} ASSISTANT:{prompt_prefix}"
            prompt_slug = slugify(prompt+prompt_prefix)
            # Same preprocessing as layerwise_dla and every behavioral run:
            # RotatedRGBAImageDataset render at 0 deg (diagonal-margin canvas,
            # white composite, 336x336) instead of the raw file.
            image = RotatedRGBAImageDataset(str(image_path), [0])[0]

            cache_save_dir = os.path.join(results_dir, "cache", model_id, image_name, prompt_slug)
            head_dla_dir = os.path.join(results_dir, "plots", "head_dla", model_id, image_name, prompt_slug)
            os.makedirs(head_dla_dir, exist_ok=True)
            final_token_trajectory_dir = os.path.join(results_dir, "plots", "final_token_trajectory", model_id, image_name, prompt_slug)
            os.makedirs(final_token_trajectory_dir, exist_ok=True)

            scan_dir = os.path.join(results_dir, "plots", "sliding_window_knockout", model_id, image_name, prompt_slug)
            os.makedirs(scan_dir, exist_ok=True)

            scan_result = collect_sliding_window_text_to_image_knockout_logits(
                model=model,
                processor=processor,
                image=image,
                prompt=text_input,
                device=device,
                window_size=3,
                target_pos=-1,
                tokens_of_interest=tokens_of_interest,
                attention_knockout_stage="post_softmax",   # default recommended
                img_token_id=32000,
            )

            plot_sliding_window_knockout_effect(
                scan_result=scan_result,
                tokens_of_interest=tokens_of_interest,
                pos_index=0,
                use_delta=True,
                save_path=os.path.join(scan_dir, "final_token_delta.png"),
                title=f"({model_id})\nQ:{prompt} A:{prompt_prefix}\nSliding-window knockout: text→image attention removed",
            )
            # plot_sliding_window_knockout_effect(
            #     # show_diff=[(0, 1)],   # e.g. bird - rabb

            np.save(os.path.join(scan_dir, "scan_window_logits.npy"), scan_result["window_logits"])
            np.save(os.path.join(scan_dir, "scan_baseline_logits.npy"), scan_result["baseline_logits"])
            with open(os.path.join(scan_dir, "scan_windows.json"), "w") as f:
                json.dump({"windows": scan_result["windows"]}, f)

            window_vis_dir = os.path.join(results_dir, "plots", "window_attention", model_id, image_name, prompt_slug)
            os.makedirs(window_vis_dir, exist_ok=True)

            windows = [(0, 15), (16, 20), (21, 25), (26, 31)]

            # raw attention
            show_window_mean_image_attention(
                model=model,
                processor=processor,
                image=image,
                prompt=text_input,
                windows=windows,
                device=device,
                mode="raw_attention",
                extra_query_pos=-10,
                img_token_id=32000,
                cache_save_dir=os.path.join(cache_save_dir, "window_attn_cache"),
                save_path=os.path.join(window_vis_dir, "raw_attention_windows.png"),
                title=f"({model_id})\nQ:{prompt} A:{prompt_prefix}\nWindowed mean raw attention to image tokens",
            )

            # contribution-like map (not direct logit attribution)
            show_window_mean_image_attention(
                model=model,
                processor=processor,
                image=image,
                prompt=text_input,
                windows=windows,
                device=device,
                mode="contribution",
                extra_query_pos=-10,
                img_token_id=32000,
                cache_save_dir=os.path.join(cache_save_dir, "window_attn_cache"),
                save_path=os.path.join(window_vis_dir, "contribution_windows.png"),
                title=f"({model_id})\nQ:{prompt} A:{prompt_prefix}\nWindowed mean image contribution (attention × ||OV(value)||)",
            )
            
            # continue

            original_cache = get_cache(
                model, processor, image, text_input, device=device,
                attention_knockout_layer=None,
                attention_knockout_mode=None,
                img_token_id=32000,
                save_dir=os.path.join(cache_save_dir, f"no_knockout")
            )
            original_logits = logitlens(
                model, original_cache["resid"], tokens_of_interest, 
                pos=[t for t in range(original_cache["resid"].shape[-2])],
                ln_mode="cached",
                ln_ref=original_cache["resid_post"][-1],
                device=device, 
                save_path=os.path.join(cache_save_dir, "original_logitlens.npy")
            )
            
            # DLA of each head
            for keep_only_key in ["image", "query"]:
                attention_knockout_query = "last"
                attention_knockout_stage = "post_softmax"
                logits_heads_knockout = collect_knockout_head_out_dla_by_layer( # [L,H,P,T]
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=text_input,
                    device=device,
                    tokens_of_interest=tokens_of_interest,
                    pos=[t for t in range(original_cache["resid"].shape[-2])],
                    attention_knockout_key=keep_only_key,
                    attention_knockout_query=attention_knockout_query,
                    attention_knockout_mode="keep_only", # drop or keep_only
                    attention_knockout_stage=attention_knockout_stage, # pre_softmax or post_softmax
                    save_path=os.path.join(cache_save_dir, f"{keep_only_key}_{attention_knockout_stage}_head_out_dla.npy")
                )
                os.makedirs(os.path.join(head_dla_dir, f"{keep_only_key}_{attention_knockout_stage}"), exist_ok=True)
                # show_logitlens_of_heads(
                #     logits_heads_knockout[..., -1, :], # [L,H,d]
                #     tokens_of_interest,

                if keep_only_key == "image":
                    show_weighted_image_attention_by_dla(
                        tokenizer,
                        text_input,
                        image,
                        attn_map=original_cache["attn_map"],                 # NEW: true post-softmax attention
                        logits_heads=logits_heads_knockout[..., -1, :],
                        tokens_of_interest=tokens_of_interest,
                        show_diff=[(0, 1)],
                        layers=list(range(15, 32)),
                        query_pos=-1,
                        top_k=4,
                        every_k_layers=6,
                        save_dir=os.path.join(head_dla_dir, "exact_attn_by_dla", f"{keep_only_key}_{attention_knockout_stage}"),

                        # NEW exact-mode args
                        model=model,
                        vproj_out=original_cache["vproj_out"],
                        resid_post_ref=original_cache["resid_post"][-1],
                    )

                    show_weighted_image_attention_by_dla(
                        tokenizer,
                        text_input,
                        image,
                        attn_map=original_cache["attn_map"],
                        logits_heads=logits_heads_knockout[..., -1, :],
                        tokens_of_interest=tokens_of_interest,
                        show_diff=[(0, 1)],
                        layers=list(range(15, 32)),
                        query_pos=-1,
                        top_k=6,
                        every_k_layers=6,
                        save_dir=os.path.join(head_dla_dir, "exact_attn_by_dla_all_summed", f"{keep_only_key}_{attention_knockout_stage}"),

                        model=model,
                        vproj_out=original_cache["vproj_out"],
                        resid_post_ref=original_cache["resid_post"][-1],
                    )
                    
            
            continue
            # plot layer trajectory
            attention_knockout_stage = "pre_softmax"
            attention_knockout_mode = "drop"
            for attention_knockout_layer in range(len(model.language_model.model.layers)):
                for attention_knockout_key in ["image", "query"]:
                    for attention_knockout_query in ["last", "all"]:
                        cache_save_dir = os.path.join(results_dir, "cache", f"{image_name}_{prompt_slug}_{model_id}_AK_L{attention_knockout_layer}_{attention_knockout_mode}_{attention_knockout_key}_to_{attention_knockout_query}")
                        os.makedirs(cache_save_dir, exist_ok=True)
                        cache = get_cache(
                            model, processor, image, prompt, device=device,
                            save_dir=None,
                            attention_knockout_layer=attention_knockout_layer,
                            attention_knockout_key=attention_knockout_key,
                            attention_knockout_query=attention_knockout_query,
                            attention_knockout_stage=attention_knockout_stage,
                            attention_knockout_mode=attention_knockout_mode,
                            img_token_id=32000,
                            knockout_value="-inf",  # or "finfo_min" if fp16 gets weird
                        )
                        for component in ["resid"]:
                            logits_after_ak = logitlens(
                                model, cache[component], tokens_of_interest, 
                                pos=[t for t in range(cache[component].shape[-2])], 
                                device=device, 
                            )
                            x_token = "▁du"
                            y_token = "▁rabb"
                            plot_layer_trajectory(
                                logits_after_ak[..., -1, :],
                                tokens_of_interest=tokens_of_interest,
                                original_logits=original_logits[..., -1, :],
                                x_token=x_token,
                                y_token=y_token,
                                image=image,
                                title=f"({model_id})\n{prompt}\nAttntion knockout L{attention_knockout_layer} ({attention_knockout_mode}: {attention_knockout_key} to {attention_knockout_query})",
                                save_path=os.path.join(final_token_trajectory_dir, f"{x_token}_{y_token}_AK_L{attention_knockout_layer}_{attention_knockout_mode}_{attention_knockout_key}_to_{attention_knockout_query}.png"),
                            )
        

            
        model.to("cpu")
        del model, processor
        torch.cuda.empty_cache()