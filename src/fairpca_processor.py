"""FairPCA processor for bias removal in text-to-image models.

Implements embedding modification using Fair PCA to remove demographic bias
while preserving image generation quality.
"""

from itertools import product

import torch

from base_processor import BaseProcessor
from fair_PCA import FairKernelPCA, FairPCA


class FairPCAProcessor(BaseProcessor):
    """Processor that applies FairPCA transformations to embeddings."""

    def modify_embedding(
        self,
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        usermode=None,
        exp_dir=".",
    ):
        """Apply FairPCA transformation to remove demographic bias from embeddings."""
        if usermode is None:
            usermode = {}
        if "remove" in usermode:

            if not hasattr(pipe, "fpca"):
                pipe.fpcas = calc_projection_matrix(exp_dir, usermode)
                for _i, fpca in enumerate(pipe.fpcas):
                    fpca.UUT = torch.tensor(
                        fpca.UUT,
                        dtype=pooled_prompt_embeds.dtype,
                    ).to(
                        pooled_prompt_embeds.device,
                    )
                    fpca.nzTXT = torch.tensor(
                        fpca.nzTXT,
                        dtype=pooled_prompt_embeds.dtype,
                    ).to(
                        pooled_prompt_embeds.device,
                    )
            else:
                pass  # Using cached FairPCA projection matrices

            # Check encoder type and handle dimensions accordingly
            encoder_type = usermode.get("encoder", "default")
            embed_dim = prompt_embeds.shape[-1]
            pooled_prompt_embeds.shape[-1]

            if embed_dim == 2048:  # SDXL case
                # SDXL has 768 (text_encoder) + 1280 (text_encoder_2)
                prompt_embeds1, prompt_embeds2 = (
                    prompt_embeds[:, :, :768],
                    prompt_embeds[:, :, 768:],
                )
            elif encoder_type.lower().startswith("t5"):
                # For T5: The encoder wrapper projects T5 768-dim to SD3 4096-dim
                # But the features were extracted with pure T5 768-dim
                # We need to work with the actual runtime dimensions
                uut_dim = len(pipe.fpcas[0].UUT)
                if embed_dim >= uut_dim:
                    # Use the runtime embedding size for processing
                    prompt_embeds1, prompt_embeds2 = (
                        prompt_embeds[:, :, :uut_dim],
                        prompt_embeds[:, :, uut_dim:],
                    )
                else:
                    # Fallback: use full embedding
                    prompt_embeds1 = prompt_embeds
                    prompt_embeds2 = torch.zeros_like(
                        prompt_embeds[:, :, :0],
                    )  # Empty tensor
            else:
                # SD3 case or others - split based on UUT dimension
                uut_dim = len(pipe.fpcas[0].UUT)
                prompt_embeds1, prompt_embeds2 = (
                    prompt_embeds[:, :, :uut_dim],
                    prompt_embeds[:, :, uut_dim:],
                )

            if "renorm" in usermode:
                pooled_prompt_embeds_norm = pooled_prompt_embeds.norm(
                    dim=1,
                    keepdim=True,
                )
                prompt_embeds1_norm, prompt_embeds2_norm = prompt_embeds1.norm(
                    dim=1,
                    keepdim=True,
                ), prompt_embeds2.norm(dim=1, keepdim=True)

            # Apply transformations
            for _i, fpca in enumerate(pipe.fpcas):
                pooled_prompt_embeds.clone()
                pooled_prompt_embeds = fpca.transform(pooled_prompt_embeds)

                # Only transform if dimensions match
                if prompt_embeds1.shape[-1] == fpca.UUT.shape[0]:
                    prompt_embeds1.clone()
                    prompt_embeds1 = fpca.transform(prompt_embeds1)
                else:
                    pass

                if prompt_embeds2.shape[-1] == fpca.UUT.shape[0]:
                    prompt_embeds2.clone()
                    prompt_embeds2 = fpca.transform(prompt_embeds2)
                else:
                    pass
            if "renorm" in usermode:
                pooled_prompt_embeds = (
                    pooled_prompt_embeds
                    / pooled_prompt_embeds.norm(dim=1, keepdim=True)
                    * pooled_prompt_embeds_norm
                )
                prompt_embeds1 = (
                    prompt_embeds1
                    / prompt_embeds1.norm(dim=1, keepdim=True)
                    * prompt_embeds1_norm
                )
                prompt_embeds2 = (
                    prompt_embeds2
                    / prompt_embeds2.norm(dim=1, keepdim=True)
                    * prompt_embeds2_norm
                )

            prompt_embeds = torch.cat((prompt_embeds1, prompt_embeds2), dim=-1)

        return prompt_embeds, pooled_prompt_embeds


def calc_projection_matrix(exp_dir, usermode):
    """Calculate FairPCA projection matrix from extracted features."""
    # Use encoder-specific feature file
    encoder_type = usermode.get("encoder", "default")
    feature_filename = f"extracted_features_{encoder_type}.pt"
    data = torch.load(f"{exp_dir}/{feature_filename}")
    tradeoff = usermode.get("tradeoff", 0.4)  # No use
    hdim = usermode.get("hdim", 600)
    if "cross" in usermode:
        fpcas = [FairPCA(target_dim=hdim, standardize=False, tradeoff_param=tradeoff)]
    elif "kernel" not in usermode:
        fpcas = [
            FairPCA(target_dim=hdim, standardize=False, tradeoff_param=tradeoff)
            for _ in range(len(data))
        ]
    else:
        fpcas = [
            FairKernelPCA(
                target_dim=hdim,
                kernel="rbf",
                degree_kernel=2,
                gamma_kernel="auto",
                standardize=False,
                tradeoff_param=tradeoff,
            )
            for _ in range(len(data))
        ]

    for fpca in fpcas:
        fpca.usermode = usermode
    if len(data.keys()) == 1:
        protect = next(iter(data.keys()))
        if "rndsample" in usermode:
            calc_projection_matrix_rndsample(data, fpca)
        if len(data[protect].keys()) == 2:
            calc_projection_matrix_sg(data[protect], fpca)
        else:
            calc_projection_matrix_mg(data[protect], fpca)
    elif "cross" in usermode:
        calc_projection_matrix_mgmd_cross(data, fpca)
    else:
        calc_projection_matrix_mgmd(data, fpcas)
    for fpca, protect in zip(fpcas, data, strict=False):
        fpca.get_emperical(data, usermode)
        fpca.protect = protect
    return fpcas


def calc_projection_matrix_sg(data, fpca) -> None:
    """Calculate projection matrix for single group (binary protected attribute)."""
    keys = list(data.keys())
    X = torch.cat(list(data.values()), dim=0)
    z = torch.cat((torch.zeros(len(data[keys[0]])), torch.ones(len(data[keys[1]])))).to(
        X.dtype,
    )
    fpca.fit(X, z)


def calc_projection_matrix_mg(data, fpca) -> None:
    """Calculate projection matrix for multiple groups (multi-class protected attribute)."""
    keys = list(data.keys())
    X = torch.cat(list(data.values()), dim=0)
    Z = torch.zeros(X.shape[0], len(keys)).type_as(X)
    st = 0
    for i, k in enumerate(keys):
        Z[st : st + data[k].shape[0], i] = 1.0
        st += data[k].shape[0]
    fpca.fit_mg(X, Z)


def calc_projection_matrix_mgmd(data, fpcas) -> None:
    """Calculate projection matrices for multiple groups, multiple demographics."""
    for protect, fpca in zip(data, fpcas, strict=False):
        keys = list(data[protect].keys())
        X = torch.cat(list(data[protect].values()), dim=0)
        Z = torch.zeros(X.shape[0], len(keys)).type_as(X)
        st = 0
        for i, k in enumerate(keys):
            Z[st : st + data[protect][k].shape[0], i] = 1.0
            st += data[protect][k].shape[0]
        fpca.fit_mg(X, Z)


def calc_projection_matrix_mgmd_cross(data, fpca) -> None:
    """Calculate projection matrix using cross-product of all demographic groups."""
    Xs, Zs = [], []
    gid = 0
    cross = list(product(*[data[protect].keys() for protect in data]))
    for gid, comb in enumerate(cross):
        xss, zss = [], []
        for pi, protect in enumerate(data):
            xss.append(data[protect][comb[pi]])
            zss.append(torch.ones(data[protect][comb[pi]].shape[0]) * gid)
        Xs.append(torch.cat(xss, dim=0))
        Zs.append(torch.cat(zss, dim=0))
    X = torch.cat(Xs, dim=0)
    Zid = torch.cat(Zs, dim=0).long()
    Z = torch.zeros(X.shape[0], gid + 1).type_as(X)
    Z[torch.arange(X.shape[0]), Zid] = 1.0
    fpca.fit_mg(X, Z)


def calc_projection_matrix_rndsample(data, fpca) -> None:
    """Calculate projection matrix with random sampling for robustness."""
    Xs, Zs = [], []
    splitcnt = fpca.usermode["rndsample"] or 5
    for protect in data:
        keys = list(data[protect].keys())
        X = torch.cat(list(data[protect].values()), dim=0)
        Z = torch.zeros(X.shape[0], len(keys)).type_as(X)
        st = 0
        for i, k in enumerate(keys):
            Z[st : st + data[protect][k].shape[0], i] = 1.0
            st += data[protect][k].shape[0]
        # Shuffle rows of X and Z
        indices = torch.randperm(X.shape[0])
        X = X[indices]
        Z = Z[indices]

        # Split into splitcnt
        split_size = X.shape[0] // splitcnt
        Xs_split = torch.split(X, split_size)
        Zs_split = torch.split(Z, split_size)

        # Append to Xs and Zs
        Xs.extend(Xs_split)
        Zs.extend(Zs_split)
    fpca.fit_mgmd(Xs, Zs)
