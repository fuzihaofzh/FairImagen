import torch
import os

class BaseProcessor():
    def __init__(self):
        self.feature_file_name = "extracted_features.pt"
        
    def get_feature_filename(self, usermode):
        """Get feature filename based on encoder type"""
        encoder_type = usermode.get("encoder", "default")
        base_name = self.feature_file_name.replace(".pt", "")
        return f"{base_name}_{encoder_type}.pt"

    def extract_embedding(self, prompt_embeds, pooled_prompt_embeds, processor, usermode={}, imagemodel = "sd3.0", exp_dir = ".", **kwargs):
        tensor = pooled_prompt_embeds[0].cpu()
        directory = exp_dir
        # Use encoder-specific filename
        feature_filename = processor.get_feature_filename(usermode)
        file_path = os.path.join(directory, feature_filename)

        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # If the file already exists, load it first and then append the new tensor
        if os.path.exists(file_path):
            data = torch.load(file_path)
        else:
            data = {}
        if kwargs['protect'] not in data:
            data[kwargs['protect']] = {}
        data[kwargs['protect']][kwargs['cat']] = torch.cat((data[kwargs['protect']].get(kwargs['cat'], torch.tensor([])), tensor.unsqueeze(0)), dim=0)

        if "wordemb" in usermode:
            emb = prompt_embeds.reshape([-1, prompt_embeds.shape[-1]])
            emb1, emb2 = emb[:, :pooled_prompt_embeds.shape[1]].cpu(), emb[:, pooled_prompt_embeds.shape[1]:].cpu()
            data[kwargs['protect']][kwargs['cat']] = torch.cat([data[kwargs['protect']][kwargs['cat']], emb1, emb2])

        # Save to file
        torch.save(data, file_path)

    def modify_embedding(self, pipe, prompt_embeds, pooled_prompt_embeds, usermode = {}, exp_dir = "."):
        return prompt_embeds, pooled_prompt_embeds

    def modify_prompt(self, prompt, usermode, num_images):
        return prompt

    def process_input(self, prompts, usermode, protect):
        return prompts