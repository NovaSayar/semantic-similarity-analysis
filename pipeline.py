# Improved Text-Image-Text-Image Pipeline Study
# Main pipeline class

import gc
import pandas as pd
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score
import clip
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SemanticSimilarityPipeline:
    def __init__(self, device='auto'):
        """Initialize the complete pipeline with proper error handling"""
        # Device configuration
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.load_models()
        
    def load_models(self):
        """Load all models with proper error handling"""
        try:
            # Stable Diffusion 2.1
            print("Loading Stable Diffusion 2.1...")
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                safety_checker=None,  # Disable for research
                requires_safety_checker=False
            ).to(self.device)
            
            # Memory optimization
            self.sd_pipeline.enable_xformers_memory_efficient_attention()
            if hasattr(self.sd_pipeline, 'enable_model_cpu_offload'):
                self.sd_pipeline.enable_model_cpu_offload()
            if hasattr(self.sd_pipeline, 'enable_model_cpu_offload'):
                self.sd_pipeline.enable_model_cpu_offload()
            
            # BLIP-2
            print("Loading BLIP-2...")
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=self.torch_dtype,
                device_map="auto"
            )
            
            # SentenceBERT
            print("Loading SentenceBERT...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # CLIP
            print("Loading CLIP...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Verify CLIP preprocessing consistency
            print(f"CLIP preprocessing normalization: {self.clip_preprocess.transforms[-1].mean}, {self.clip_preprocess.transforms[-1].std}")
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def generate_image(self, prompt, seed=None, steps=30, guidance_scale=7.5):
        """Generate image with better parameters and error handling"""
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
            
            # Better parameters for quality
            image = self.sd_pipeline(
                prompt,
                num_inference_steps=steps,  # Increased for better quality
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,  # Explicit size
                width=512
            ).images[0]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return image
            
        except Exception as e:
            print(f"Error generating image for prompt '{prompt[:50]}...': {e}")
            return None
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """Generate caption with improved parameters"""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,  # More beams for better quality
                    do_sample=False,
                    repetition_penalty=1.1,  # Reduce repetition
                    length_penalty=1.0
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            return caption.strip()
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None
    
    def complete_pipeline(self, original_prompt, category, prompt_type, prompt_id, seed=42, save_images=True):
        """Complete pipeline with enhanced logging and saving"""
        print(f"\n{'='*80}")
        print(f"Processing {category.upper()} - {prompt_type.upper()} - ID: {prompt_id}")
        print(f"Original prompt (C1): {original_prompt}")
        print(f"Seed: {seed}")
        print(f"{'='*80}")
        
        # Create results directory
        if save_images:
            # Create images folder if it doesn't exist
            os.makedirs('images', exist_ok=True)
            results_dir = f"images/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(results_dir, exist_ok=True)
        
        # Stage 1: C1 → I1
        print("Stage 1: C1 → I1 (Text to Image)")
        image1 = self.generate_image(original_prompt, seed)
        
        if image1 is None:
            print("Failed at Stage 1")
            return None
        
        if save_images:
            image1.save(f"{results_dir}/I1_{category}_{prompt_type}_{prompt_id}_seed{seed}.png")
        
        # Stage 2: I1 → C2
        print("Stage 2: I1 → C2 (Image to Text)")
        caption2 = self.generate_caption(image1)
        
        if caption2 is None:
            print("Failed at Stage 2")
            return None
        
        print(f"Generated caption (C2): {caption2}")
        
        # Stage 3: C2 → I2
        print("Stage 3: C2 → I2 (Text to Image)")
        image2 = self.generate_image(caption2, seed)
        
        if image2 is None:
            print("Failed at Stage 3")
            return None
        
        if save_images:
            image2.save(f"{results_dir}/I2_{category}_{prompt_type}_{prompt_id}_seed{seed}.png")
        
        # Display images
        self.display_images(image1, image2, category, prompt_type, original_prompt, caption2)
        
        return {
            'image1': image1,
            'image2': image2,
            'caption2': caption2,
            'results_dir': results_dir if save_images else None
        }
    
    def display_images(self, image1, image2, category, prompt_type, original_prompt, caption):
        """Display images with better formatting"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(image1)
        axes[0].set_title(f"I1: {category}-{prompt_type}\n{original_prompt[:50]}...", 
                         fontsize=10, wrap=True)
        axes[0].axis('off')
        
        axes[1].imshow(image2)
        axes[1].set_title(f"I2: Reconstructed\n{caption[:50]}...", 
                         fontsize=10, wrap=True)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
