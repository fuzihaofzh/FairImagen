"""
Pipeline Factory for Fair Image Generation
Supports latest open-source text-to-image models (2024-2025)
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline, 
    StableDiffusion3Pipeline,
    DiffusionPipeline
)
from sdpipline import UserStableDiffusion3Pipeline
from sdxl_pipeline_new import UserStableDiffusionXLPipeline
from pipeline_adapters import (
    UserFluxPipeline,
    UserPixArtSigmaPipeline,
    UserWuerstchenPipeline,
    create_adapted_pipeline
)

def create_pipeline(pipename="sd3.0", **kwargs):
    """
    Factory function to create different types of pipelines
    
    Supported open-source pipelines (2024-2025):
    
    Stable Diffusion Family:
    - sd3.0: Stable Diffusion 3.0 Medium (2B params)
    - sd3.5: Stable Diffusion 3.5 Medium  
    - sdxl: Stable Diffusion XL 1.0
    - sdxl-turbo: SDXL Turbo (fast variant)
    - sdxl-lightning: SDXL Lightning (optimized for speed)
    
    FLUX Family (Black Forest Labs - from SD creators):
    - flux-schnell: FLUX.1 schnell (fast, open source)
    - flux-dev: FLUX.1 dev (research only)
    
    Other Notable Models:
    - playground-v2.5: Playground 2.5 (Midjourney-style)
    - pixart-sigma: PixArt-Σ (DiT architecture)
    - wurstchen: Würstchen v3 (efficient cascade)
    """
    
    # Stable Diffusion 3.x series (2024)
    if pipename == "sd3.5":
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        pipe = UserStableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        )
    elif pipename == "sd3.0":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        pipe = UserStableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
    
    # Stable Diffusion XL variants
    elif pipename == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = UserStableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    elif pipename == "sdxl-turbo":
        model_id = "stabilityai/sdxl-turbo"
        pipe = UserStableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    elif pipename == "sdxl-lightning":
        # SDXL Lightning - 4-step generation
        model_id = "ByteDance/SDXL-Lightning"
        pipe = UserStableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        # Would need to load LoRA weights for Lightning
    
    # FLUX series (2024 - from SD creators at Black Forest Labs)
    elif pipename == "flux-schnell":
        try:
            # FLUX.1 schnell - fully open source, Apache 2.0
            model_id = "black-forest-labs/FLUX.1-schnell"
            pipe = UserFluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
        except:
            # Fallback if FLUX not available in diffusers yet
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                custom_pipeline="flux"
            )
    
    elif pipename == "flux-dev":
        try:
            # FLUX.1 dev - open weights, research only
            model_id = "black-forest-labs/FLUX.1-dev"
            pipe = UserFluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
        except:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                custom_pipeline="flux"
            )
    
    # Playground (Midjourney-style aesthetic)
    elif pipename == "playground-v2.5":
        model_id = "playgroundai/playground-v2.5-1024px-aesthetic"
        # Playground is based on SDXL, so use SDXL adapter
        pipe = UserStableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    
    # PixArt-Σ (DiT architecture, like SD3)
    elif pipename == "pixart-sigma":
        try:
            from diffusers import PixArtSigmaPipeline
            model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
            base_pipe = PixArtSigmaPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            # Use our custom adapter for PixArt
            pipe = UserPixArtSigmaPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        except ImportError:
            raise ValueError("PixArt-Sigma requires latest diffusers")
    
    # Würstchen v3 (efficient cascade model)
    elif pipename == "wurstchen":
        try:
            from diffusers import WuerstchenDecoderPipeline
            model_id = "warp-ai/wuerstchen"
            base_pipe = WuerstchenDecoderPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
            # Use our custom Würstchen adapter
            pipe = UserWuerstchenPipeline(base_pipe)
        except ImportError:
            raise ValueError("Würstchen requires latest diffusers")
    
    else:
        raise ValueError(f"Unknown pipeline: {pipename}")
    
    return pipe

# Utility functions
def get_supported_pipelines():
    """Return list of supported pipeline names"""
    return [
        # Stable Diffusion Family
        "sd3.0", "sd3.5", "sdxl", "sdxl-turbo", "sdxl-lightning",
        
        # FLUX Family (from SD creators)
        "flux-schnell", "flux-dev",
        
        # Other Notable Models
        "playground-v2.5", "pixart-sigma", "wurstchen"
    ]

def get_pipeline_info(pipename):
    """Get information about a specific pipeline"""
    info = {
        # Stable Diffusion 3.x
        "sd3.0": {
            "type": "text2img", 
            "arch": "DiT (Diffusion Transformer)", 
            "params": "2B",
            "size": "1024x1024", 
            "speed": "medium",
            "year": "2024"
        },
        "sd3.5": {
            "type": "text2img", 
            "arch": "DiT (Diffusion Transformer)", 
            "params": "2.5B",
            "size": "1024x1024", 
            "speed": "medium",
            "year": "2024"
        },
        
        # SDXL Family
        "sdxl": {
            "type": "text2img", 
            "arch": "UNet", 
            "params": "3.5B",
            "size": "1024x1024", 
            "speed": "slow",
            "year": "2023"
        },
        "sdxl-turbo": {
            "type": "text2img", 
            "arch": "UNet (distilled)", 
            "params": "3.5B",
            "size": "512x512", 
            "speed": "very_fast",
            "year": "2023"
        },
        "sdxl-lightning": {
            "type": "text2img", 
            "arch": "UNet + LoRA", 
            "params": "3.5B",
            "size": "1024x1024", 
            "speed": "very_fast", 
            "year": "2024"
        },
        
        # FLUX Family
        "flux-schnell": {
            "type": "text2img", 
            "arch": "Flow Matching", 
            "params": "12B",
            "size": "1024x1024", 
            "speed": "fast (1-4 steps)",
            "year": "2024",
            "license": "Apache 2.0"
        },
        "flux-dev": {
            "type": "text2img", 
            "arch": "Flow Matching", 
            "params": "12B",
            "size": "1024x1024", 
            "speed": "medium",
            "year": "2024",
            "license": "Research only"
        },
        
        # Others
        "playground-v2.5": {
            "type": "text2img", 
            "arch": "SDXL-based", 
            "params": "3.5B",
            "size": "1024x1024", 
            "speed": "medium",
            "year": "2024",
            "note": "Midjourney aesthetic"
        },
        "pixart-sigma": {
            "type": "text2img", 
            "arch": "DiT", 
            "params": "0.6B",
            "size": "1024x1024", 
            "speed": "fast",
            "year": "2024"
        },
        "wurstchen": {
            "type": "text2img", 
            "arch": "Cascade (Stage C+B)", 
            "params": "1B+0.3B",
            "size": "1024x1024", 
            "speed": "fast",
            "year": "2024"
        }
    }
    return info.get(pipename, {"type": "unknown", "arch": "unknown", "params": "unknown"})