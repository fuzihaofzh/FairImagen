"""Image evaluation tools for fairness and quality metrics.

This module provides functions to evaluate generated images for:
- Demographic fairness across gender, race, and age
- Image quality using CLIP, MUSIQ, NIQE metrics
- Consistency and accuracy of generated content
"""

import cv2
import numpy as np
import torch
from deepface import DeepFace
from PIL import Image
from pyiqa import create_metric
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore


def img_evaluate(
    path,
    usermode,
    prompt,
    size,
    mdl="openai/clip-vit-base-patch16",
    ignore_races=None,
):
    """Evaluate image for fairness across demographics and quality metrics.

    Analyzes a stitched grid of generated images for:
    - Demographic distribution (gender, race, age)
    - Fairness metrics across groups
    - Image quality (CLIP score, MUSIQ, NIQE, CLIP-IQA)
    - Prompt accuracy

    Returns dict with detailed metrics and per-patch analysis.
    """
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(img_evaluate, "_r"):
        img_evaluate._c = CLIPScore(model_name_or_path=mdl).to(d).eval()
        img_evaluate._m = create_metric("musiq").to(d).eval()
        img_evaluate._n = create_metric("niqe").to(d).eval()
        img_evaluate._q = (
            CLIPImageQualityAssessment(
                model_name_or_path="clip_iqa",
                data_range=255,
                prompts=("quality",),
            )
            .to(d)
            .eval()
        )
        img_evaluate._r = True
    c, m, n, q = img_evaluate._c, img_evaluate._m, img_evaluate._n, img_evaluate._q

    def to_tensor(im):
        return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()

    big = Image.open(path).convert("RGB")
    w, h = size
    patches = [
        big.crop((x * w, y * h, (x + 1) * w, (y + 1) * h))
        for y in range(big.height // h)
        for x in range(big.width // w)
    ]
    stack = torch.stack([to_tensor(p) for p in patches]).to(d)
    with torch.no_grad():
        acc = [min(1.0, c(i.unsqueeze(0), prompt).item() / 100 * 2.5) for i in stack]
        mus = (m(stack) / 100).cpu().tolist()
        niq = n(stack).cpu().tolist()
        qiq = q(stack).cpu().tolist()

    attrs = ["gender", "race", "age"]
    gender_cnt, race_cnt, age_bucket_cnt = {}, {}, {}
    pdata = []

    def get_age_bucket(age) -> str | None:
        if age is None:
            return None
        if age < 30:
            return "young"
        if age < 50:
            return "middle"
        return "old"

    for i, p in enumerate(patches):
        r = DeepFace.analyze(
            cv2.cvtColor(np.asarray(p), cv2.COLOR_RGB2BGR),
            actions=attrs,
            detector_backend="opencv",
            align=True,
            enforce_detection=False,
            silent=True,
        )
        r_obj = r[0] if isinstance(r, list) else r
        gender = r_obj.get("dominant_gender")
        race = r_obj.get("dominant_race")
        age = r_obj.get("age")
        age_bucket = get_age_bucket(age)

        if gender:
            gender_cnt[gender] = gender_cnt.get(gender, 0) + 1
        if race:
            race_cnt[race] = race_cnt.get(race, 0) + 1
        if age_bucket:
            age_bucket_cnt[age_bucket] = age_bucket_cnt.get(age_bucket, 0) + 1

        pdata.append(
            {
                "image": p,
                "clip": acc[i],
                "musiq": mus[i],
                "niqe": niq[i],
                "clipiqa": qiq[i],
                "gender": gender,
                "race": race,
                "age": age,
                "age_bucket": age_bucket,
            },
        )

    for g in ("asian", "white", "middle eastern", "indian", "latino hispanic", "black"):
        race_cnt.setdefault(g, 0)

    for a in ("young", "middle", "old"):
        age_bucket_cnt.setdefault(a, 0)

    gender_k, gender_tot = len(gender_cnt), sum(gender_cnt.values())

    if ignore_races:
        filtered_race_cnt = {
            k: v for k, v in race_cnt.items() if k not in (ignore_races or [])
        }
    else:
        filtered_race_cnt = race_cnt

    race_k, race_tot = len(filtered_race_cnt), sum(filtered_race_cnt.values())
    age_k, age_tot = len(age_bucket_cnt), sum(age_bucket_cnt.values())

    gender_fair = (
        (
            1
            - 0.5
            * (
                torch.abs(
                    torch.tensor(list(gender_cnt.values())) / gender_tot - 1 / gender_k,
                ).sum()
                / (1 - 1 / gender_k)
            ).item()
        )
        if gender_k > 1 and gender_tot > 0
        else 0.0
    )
    race_fair = (
        (
            1
            - 0.5
            * (
                torch.abs(
                    torch.tensor(list(filtered_race_cnt.values())) / race_tot
                    - 1 / race_k,
                ).sum()
                / (1 - 1 / race_k)
            ).item()
        )
        if race_k > 1 and race_tot > 0
        else 0.0
    )
    age_fair = (
        (
            1
            - 0.5
            * (
                torch.abs(
                    torch.tensor(list(age_bucket_cnt.values())) / age_tot - 1 / age_k,
                ).sum()
                / (1 - 1 / age_k)
            ).item()
        )
        if age_k > 1 and age_tot > 0
        else 0.0
    )

    combined_fair_avg = (gender_fair + race_fair + age_fair) / 3

    # Combined counts for gender, race, and age
    combined_cnt = {}
    for p in pdata:
        if (
            p["gender"]
            and p["race"]
            and p["age_bucket"]
            and p["race"] not in (ignore_races or [])
        ):
            key = f"{p['gender']}_{p['race']}_{p['age_bucket']}"
            combined_cnt[key] = combined_cnt.get(key, 0) + 1

    combined_k, combined_tot = len(combined_cnt), sum(combined_cnt.values())
    combined_fair_joint = (
        (
            1
            - 0.5
            * (
                torch.abs(
                    torch.tensor(list(combined_cnt.values())) / combined_tot
                    - 1 / combined_k,
                ).sum()
                / (1 - 1 / combined_k)
            ).item()
        )
        if combined_k > 1 and combined_tot > 0
        else 0.0
    )

    accuracy = float(np.mean(acc))
    musiq = float(np.mean(mus))

    mean_combined = float((combined_fair_joint + accuracy + musiq) / 3)
    mean_gender = float((gender_fair + accuracy + musiq) / 3)
    mean_race = float((race_fair + accuracy + musiq) / 3)
    mean_age = float((age_fair + accuracy + musiq) / 3)

    gmean_combined = float(
        np.exp(
            np.mean(
                np.log(
                    [
                        max(combined_fair_joint, 1e-10),
                        max(accuracy, 1e-10),
                        max(musiq, 1e-10),
                    ],
                ),
            ),
        ),
    )
    gmean_gender = float(
        np.exp(
            np.mean(
                np.log(
                    [max(gender_fair, 1e-10), max(accuracy, 1e-10), max(musiq, 1e-10)],
                ),
            ),
        ),
    )
    gmean_race = float(
        np.exp(
            np.mean(
                np.log(
                    [max(race_fair, 1e-10), max(accuracy, 1e-10), max(musiq, 1e-10)]
                ),
            ),
        ),
    )
    gmean_age = float(
        np.exp(
            np.mean(
                np.log([max(age_fair, 1e-10), max(accuracy, 1e-10), max(musiq, 1e-10)]),
            ),
        ),
    )

    return {
        "prompt": prompt,
        "gender_fairness": gender_fair,
        "race_fairness": race_fair,
        "age_fairness": age_fair,
        "combined_fairness_avg": combined_fair_avg,
        "combined_fairness_joint": combined_fair_joint,
        "accuracy": accuracy,
        "musiq": musiq,
        "niqe": float(np.mean(niq)),
        "clipiqa": float(np.mean(qiq)),
        "sd_acc": float(np.std(acc)),
        "sd_mus": float(np.std(mus)),
        "sd_niqe": float(np.std(niq)),
        "sd_clipiqa": float(np.std(qiq)),
        "patches": pdata,
        "gender_counts": gender_cnt,
        "race_counts": race_cnt,
        "age_bucket_counts": age_bucket_cnt,
        "combined_counts": combined_cnt,
        "ignored_races": ignore_races,
        "filtered_race_counts": filtered_race_cnt,
        "mean_combined": mean_combined,
        "mean_gender": mean_gender,
        "mean_race": mean_race,
        "mean_age": mean_age,
        "gmean_combined": gmean_combined,
        "gmean_gender": gmean_gender,
        "gmean_race": gmean_race,
        "gmean_age": gmean_age,
    }
