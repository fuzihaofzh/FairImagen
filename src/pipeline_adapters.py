"""
Pipeline Adapters for different models to support fairness processors
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

class UserStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """SDXL Pipeline with fairness support"""
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        **kwargs
    ):
        # Call parent to get embeddings
        device = self._execution_device
        
        # Calculate do_classifier_free_guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt - SDXL has dual text encoders
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
        )
        
        # HACK POINT 1: Extract features for fairness analysis
        if hasattr(self, 'usermode') and "extract" in self.usermode:
            # SDXL uses pooled embeddings differently than SD3
            self.processor.extract_embedding(
                prompt_embeds, 
                pooled_prompt_embeds, 
                self.processor, 
                usermode=self.usermode, 
                imagemodel="sdxl", 
                exp_dir=self.exp_dir,
                **kwargs
            )
            return
        
        # Concatenate for classifier free guidance
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        # HACK POINT 2: Modify embeddings for fairness
        if hasattr(self, 'processor'):
            prompt_embeds, pooled_prompt_embeds = self.processor.modify_embedding(
                self, 
                prompt_embeds, 
                pooled_prompt_embeds, 
                usermode=self.usermode, 
                exp_dir=self.exp_dir
            )
        
        # Continue with normal SDXL pipeline
        kwargs['prompt_embeds'] = prompt_embeds
        kwargs['pooled_prompt_embeds'] = pooled_prompt_embeds
        kwargs['negative_prompt_embeds'] = None  # Already concatenated
        kwargs['negative_pooled_prompt_embeds'] = None
        
        return super().__call__(
            prompt=None,  # We already have embeddings
            prompt_2=None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=None,
            negative_prompt_2=None,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )


class UserFluxPipeline(DiffusionPipeline):
    """FLUX Pipeline with fairness support"""
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,  # FLUX schnell uses 1-4 steps
        guidance_scale: float = 0.0,  # FLUX often uses no guidance
        num_images_per_prompt: Optional[int] = 1,
        **kwargs
    ):
        # FLUX has a different architecture with larger text encoders
        # We need to handle its specific encoding process
        
        device = self._execution_device if hasattr(self, '_execution_device') else 'cuda'
        
        # FLUX specific prompt encoding
        # Note: FLUX uses T5 and CLIP encoders
        if hasattr(self, 'encode_prompt'):
            prompt_embeds = self.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            # Fallback for custom FLUX implementations
            # FLUX typically returns different embedding structure
            text_encoder_output = self.text_encoder(prompt, return_dict=True)
            prompt_embeds = text_encoder_output.last_hidden_state
            pooled_prompt_embeds = text_encoder_output.pooler_output if hasattr(text_encoder_output, 'pooler_output') else None
        
        # HACK POINT 1: Extract features
        if hasattr(self, 'usermode') and "extract" in self.usermode:
            # FLUX may not have pooled embeddings in the same way
            pooled = pooled_prompt_embeds if 'pooled_prompt_embeds' in locals() else prompt_embeds.mean(dim=1)
            self.processor.extract_embedding(
                prompt_embeds,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="flux",
                exp_dir=self.exp_dir,
                **kwargs
            )
            return
        
        # HACK POINT 2: Modify embeddings
        if hasattr(self, 'processor'):
            if 'pooled_prompt_embeds' in locals():
                prompt_embeds, pooled_prompt_embeds = self.processor.modify_embedding(
                    self,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    usermode=self.usermode,
                    exp_dir=self.exp_dir
                )
            else:
                # FLUX might only have prompt_embeds
                modified_embeds, _ = self.processor.modify_embedding(
                    self,
                    prompt_embeds,
                    prompt_embeds.mean(dim=1),  # Create pseudo pooled embeddings
                    usermode=self.usermode,
                    exp_dir=self.exp_dir
                )
                prompt_embeds = modified_embeds
        
        # Continue with FLUX generation
        kwargs['prompt_embeds'] = prompt_embeds
        
        return super().__call__(
            prompt=None,  # We already have embeddings
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )


class UserPixArtSigmaPipeline(DiffusionPipeline):
    """PixArt-Sigma Pipeline with fairness support"""
    
    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]] = None, **kwargs):
        # PixArt uses T5 encoder like SD3
        device = self._execution_device if hasattr(self, '_execution_device') else 'cuda'
        
        # Get embeddings
        if hasattr(self, 'encode_prompt'):
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                device=device,
            )
        else:
            # Fallback
            prompt_embeds = self.text_encoder(prompt)[0]
        
        # HACK POINT 1: Extract features
        if hasattr(self, 'usermode') and "extract" in self.usermode:
            # PixArt doesn't have pooled embeddings, use mean
            pooled = prompt_embeds.mean(dim=1)
            self.processor.extract_embedding(
                prompt_embeds,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="pixart",
                exp_dir=self.exp_dir,
                **kwargs
            )
            return
        
        # HACK POINT 2: Modify embeddings
        if hasattr(self, 'processor'):
            prompt_embeds, _ = self.processor.modify_embedding(
                self,
                prompt_embeds,
                prompt_embeds.mean(dim=1),
                usermode=self.usermode,
                exp_dir=self.exp_dir
            )
        
        kwargs['prompt_embeds'] = prompt_embeds
        return super().__call__(prompt=None, **kwargs)


class UserWuerstchenPipeline:
    """Würstchen Pipeline with fairness support"""
    
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        # Copy all attributes
        for attr in dir(base_pipeline):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_pipeline, attr))
    
    @torch.no_grad()
    def __call__(self, prompt=None, **kwargs):
        # Würstchen has a cascade architecture
        # We need to handle Stage C (compression) differently
        
        if hasattr(self, 'encode_prompt'):
            # Get text embeddings from prior
            text_embeddings = self.encode_prompt(prompt)
        else:
            # Direct text encoding
            text_embeddings = self.prior_text_encoder(prompt)
        
        # HACK POINT 1: Extract features
        if hasattr(self, 'usermode') and "extract" in self.usermode:
            # Würstchen doesn't have pooled embeddings
            pooled = text_embeddings.mean(dim=1)
            self.processor.extract_embedding(
                text_embeddings,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="wurstchen",
                exp_dir=self.exp_dir,
                **kwargs
            )
            return
        
        # HACK POINT 2: Modify embeddings
        if hasattr(self, 'processor'):
            text_embeddings, _ = self.processor.modify_embedding(
                self,
                text_embeddings,
                text_embeddings.mean(dim=1),
                usermode=self.usermode,
                exp_dir=self.exp_dir
            )
        
        # Continue with Würstchen generation
        kwargs['prompt_embeds'] = text_embeddings
        return self.base_pipeline(prompt=None, **kwargs)


def create_adapted_pipeline(base_pipeline, processor, usermode, exp_dir):
    """
    Inject processor and usermode into any pipeline
    This is a generic adapter that works with most pipelines
    """
    # Store processor and config in the pipeline
    base_pipeline.processor = processor
    base_pipeline.usermode = usermode
    base_pipeline.exp_dir = exp_dir
    
    # Override the __call__ method to add our hooks
    original_call = base_pipeline.__call__
    
    @torch.no_grad()
    def adapted_call(prompt=None, **kwargs):
        # Try to get embeddings using the pipeline's encode method
        if prompt is not None and hasattr(base_pipeline, 'encode_prompt'):
            # Most pipelines have encode_prompt method
            encode_result = base_pipeline.encode_prompt(prompt)
            
            # Handle different return formats
            if isinstance(encode_result, tuple):
                prompt_embeds = encode_result[0]
                pooled_embeds = encode_result[1] if len(encode_result) > 1 else prompt_embeds.mean(dim=1)
            else:
                prompt_embeds = encode_result
                pooled_embeds = prompt_embeds.mean(dim=1)
            
            # HACK POINT 1: Extract features
            if "extract" in usermode:
                processor.extract_embedding(
                    prompt_embeds,
                    pooled_embeds,
                    processor,
                    usermode=usermode,
                    imagemodel=base_pipeline.__class__.__name__,
                    exp_dir=exp_dir,
                    **kwargs
                )
                return
            
            # HACK POINT 2: Modify embeddings
            prompt_embeds, pooled_embeds = processor.modify_embedding(
                base_pipeline,
                prompt_embeds,
                pooled_embeds,
                usermode=usermode,
                exp_dir=exp_dir
            )
            
            # Update kwargs with modified embeddings
            kwargs['prompt_embeds'] = prompt_embeds
            if 'pooled_prompt_embeds' in kwargs:
                kwargs['pooled_prompt_embeds'] = pooled_embeds
            
            # Clear original prompt since we have embeddings
            prompt = None
        
        return original_call(prompt=prompt, **kwargs)
    
    # Replace the __call__ method
    base_pipeline.__call__ = adapted_call
    
    return base_pipeline