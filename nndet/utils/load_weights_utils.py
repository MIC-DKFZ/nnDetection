from einops import rearrange
import torch.nn.functional as F
import torch


def resolve_stem_override(model_cfg: dict, adaptation_plan: dict):
    """
    Returns (key_to_stem, keys_to_in_proj) for a MultiTalent-style
    checkpoint, honoring an optional model_cfg["stem_override"] that takes
    precedence over the checkpoint's own nnssl_adaptation_plan["key_to_stem"].
    Lets a single checkpoint (with several per-dataset stems) serve
    multiple stem-selection finetuning runs via a CLI override
    (`-o model_cfg.stem_override=...`) instead of duplicating or editing
    the checkpoint file -- see docs/finetuning.md 3.4.1.

    keys_to_in_proj isn't in the checkpoint under the override's key, so
    it's re-derived using the same "<key_to_stem>.convs.0.<...>" naming
    convention the checkpoint's own plan already follows.
    """
    stem_override = model_cfg.get("stem_override") or None
    if not stem_override:
        return adaptation_plan["key_to_stem"], adaptation_plan["keys_to_in_proj"]
    return stem_override, [f"{stem_override}.convs.0.conv", f"{stem_override}.convs.0.all_modules.0"]


def filter_state_dict(state_dict, skip_strings):
    found_flag = False
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if any(skip in k for skip in skip_strings):
            found_flag = True
            continue
        filtered_state_dict[k] = v

    return filtered_state_dict, found_flag

def interpolate_patch_embed_1d(patch_embed, target_len, mode="linear"):
    """Resizes patch embeddings using interpolation."""
    return F.interpolate(
        patch_embed.permute(0, 2, 1),  # [B, C, Tokens]
        size=target_len,
        mode=mode,
        align_corners=False,
    ).permute(0, 2, 1)  # [B, Tokens, C]

def interpolate_patch_embed_3d(patch_embed, in_shape, out_shape):
    """Resizes patch embeddings using 3D trilinear interpolation."""
    patch_embed = patch_embed.permute(0, 2, 1)
    patch_embed = rearrange(patch_embed, "B C (x y z) -> B C x y z", **in_shape)
    patch_embed = F.interpolate(patch_embed, size=list(out_shape.values()), mode="trilinear", align_corners=False)
    patch_embed = rearrange(patch_embed, "B C x y z -> B C (x y z)", **out_shape)
    return patch_embed.permute(0, 2, 1)

def handle_pos_embed_resize(pretrained_dict, model_dict, mode, input_shape=None, pretrained_input_patch_size=None, patch_embed_size=None):
    pretrained_pos_embed = pretrained_dict["pos_embed"]
    model_pos_embed = model_dict["pos_embed"]
    model_pos_embed_shape = model_pos_embed.shape

    has_cls_token = "cls_token" in pretrained_dict

    if has_cls_token:
        cls_pos_embed = pretrained_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]
    else:
        if  "cls_token" in model_dict.keys():
            cls_pos_embed = model_pos_embed[:, :1, :]
        patch_pos_embed = pretrained_pos_embed

    if mode == "interpolate":
        resized_patch_pos_embed = interpolate_patch_embed_1d(patch_pos_embed, target_len=model_pos_embed_shape[1] - int(has_cls_token))

    elif mode == "interpolate_trilinear":
        # Calculate input/output 3D shapes
        in_shape = dict(zip("xyz", [int(d / p) for d, p in zip(pretrained_input_patch_size, patch_embed_size)]))
        out_shape = dict(zip("xyz", [int(d / p) for d, p in zip(input_shape, patch_embed_size)]))
        resized_patch_pos_embed = interpolate_patch_embed_3d(patch_pos_embed, in_shape, out_shape)

    else:
        raise NotImplementedError(f"Unknown resize mode: {mode}")
    if "cls_token" in model_dict.keys():
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
    else:
        resized_pos_embed = resized_patch_pos_embed
    pretrained_dict["pos_embed"] = resized_pos_embed