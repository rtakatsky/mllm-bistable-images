# Llama3-LLaVA-Next answers the object-count query as "\n1": its chat template
# already ends the assistant turn with the merged token ĊĊ, so no string prefix
# reaches the position it answers from (appending "\n" re-merges to ĊĊĊ, " "
# forces an off-path Ġ). Figure 45 handles this by appending the token id for Ċ;
# this script computes the same readout for the appendix count table so figure
# and table share one setting and no unnatural prefix is introduced
# (the other four models keep the " " prefix, which is their own first token).
import glob, json, os, torch
from utils import *

MODEL = "llama3-llava-next-8b"
COUNT = "How many objects are in the image? Answer with a number only."
BASE_SLUG = slugify("List every animal in the image, each in one word." + "I see a")

if __name__ == "__main__":
    root = PROJECT_ROOT
    ddir = os.path.join(DATA_DIR, "duck_rabbit")
    boundaries = json.load(open(os.path.join(ddir, "boundaries.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model, prompt_format = load_vlm(VLM_DICT[MODEL], return_prompt_format=True)
    model = model.to(device).eval()
    tok = processor.tokenizer
    newline_id = tok.convert_tokens_to_ids("Ċ")
    assert newline_id is not None and newline_id != tok.unk_token_id
    digits = {d: tok.convert_tokens_to_ids(d) for d in "0123456789"}

    stimuli = [("harper", "harper", 0), ("raven_bear", "raven_bear", 0)]
    for f in sorted(glob.glob(os.path.join(ddir, "va_s*.png"))):
        n = os.path.basename(f)[:-4]
        a = get_boundary(boundaries, MODEL, n, BASE_SLUG)
        if a is not None:
            stimuli.append((n, "va", int(round(a))))
    valid = {n.split("_")[-1] for n, g, _ in stimuli if g == "va"}
    for f in sorted(glob.glob(os.path.join(ddir, "split_s*.png"))):
        n = os.path.basename(f)[:-4]
        if n.split("_")[-1] in valid:
            stimuli.append((n, "split", 0))
    print(f"{len(stimuli)} stimuli", flush=True)

    conv = [{"role": "user", "content": [{"type": "text", "text": COUNT}, {"type": "image"}]}]
    text = (processor.apply_chat_template(conv, add_generation_prompt=True)
            if prompt_format is None else prompt_format.format(prompt=COUNT))
    out = {"model": MODEL, "prompt": COUNT, "readout": "token-append 'Ċ' (model's own answer step)", "images": {}}
    for name, group, angle in stimuli:
        image = RotatedRGBAImageDataset(os.path.join(ddir, f"{name}.png"), [angle])[0]
        inputs = processor(images=[image], text=[text], return_tensors="pt").to(device)
        extra = torch.tensor([[newline_id]], device=device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra)], dim=1)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0, -1].float(), -1)
        rec = {d: float(probs[i]) for d, i in digits.items()}
        rec["digit_mass"] = float(sum(rec.values()))
        out["images"][name] = {"group": group, "angle": angle, "first_token": rec}
        if group in ("harper", "raven_bear"):
            print(f"  {name:12s} p1={rec['1']:.3f} p2={rec['2']:.3f} digit_mass={rec['digit_mass']:.3f}", flush=True)
    d = os.path.join(OUTPUTS_DIR, "object_counting", "tables", MODEL)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "summary_natural.json"), "w"), indent=1)
    print("done", flush=True)
