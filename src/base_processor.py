"""Base processor for fairness-aware image generation.

This module provides the base class for processing embeddings to remove or mitigate
demographic biases in text-to-image generation models.
"""

from pathlib import Path

import torch


class BaseProcessor:
    """Base processor class for embedding modification.

    Provides interface for extracting and modifying embeddings to achieve
    fairness in generated images across protected attributes.
    """

    def __init__(self) -> None:
        """Initialize the BaseProcessor."""
        self.feature_file_name = "extracted_features.pt"

    def get_feature_filename(self, usermode) -> str:
        """Get feature filename based on encoder type."""
        encoder_type = usermode.get("encoder", "default")
        base_name = self.feature_file_name.replace(".pt", "")
        return f"{base_name}_{encoder_type}.pt"

    def extract_embedding(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        processor,
        usermode=None,
        imagemodel="sd3.0",
        exp_dir=".",
        **kwargs,
    ) -> None:
        """Extract and save embeddings for fairness analysis.

        Saves embeddings from different demographic categories to analyze
        and compute fairness transformations.
        """
        if usermode is None:
            usermode = {}
        tensor = pooled_prompt_embeds[0].cpu()
        directory = Path(exp_dir)
        # Use encoder-specific filename
        feature_filename = processor.get_feature_filename(usermode)
        file_path = directory / feature_filename

        # Check if the directory exists, if not, create it
        directory.mkdir(parents=True, exist_ok=True)

        # If the file already exists, load it first and then append the new tensor
        data = torch.load(file_path) if file_path.exists() else {}
        if kwargs["protect"] not in data:
            data[kwargs["protect"]] = {}
        data[kwargs["protect"]][kwargs["cat"]] = torch.cat(
            (
                data[kwargs["protect"]].get(kwargs["cat"], torch.tensor([])),
                tensor.unsqueeze(0),
            ),
            dim=0,
        )

        if "wordemb" in usermode:
            emb = prompt_embeds.reshape([-1, prompt_embeds.shape[-1]])
            emb1, emb2 = (
                emb[:, : pooled_prompt_embeds.shape[1]].cpu(),
                emb[:, pooled_prompt_embeds.shape[1] :].cpu(),
            )
            data[kwargs["protect"]][kwargs["cat"]] = torch.cat(
                [data[kwargs["protect"]][kwargs["cat"]], emb1, emb2],
            )

        # Save to file
        torch.save(data, file_path)

    def modify_embedding(
        self,
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        usermode=None,
        exp_dir=".",
    ):
        """Modify embeddings to remove demographic bias.

        Base implementation returns embeddings unchanged. Override in subclasses
        to apply fairness transformations.
        """
        if usermode is None:
            usermode = {}
        return prompt_embeds, pooled_prompt_embeds

    def modify_prompt(self, prompt, usermode, num_images):
        """Modify prompts before embedding.

        Base implementation returns prompts unchanged. Override to apply
        prompt-level modifications.
        """
        return prompt

    def process_input(self, prompts, usermode, protect):
        """Process input prompts for feature extraction.

        Base implementation returns prompts unchanged. Override to apply
        preprocessing for specific protected attributes.
        """
        return prompts
