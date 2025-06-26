# Similarity metrics calculation

import torch
from sentence_transformers import SentenceTransformer
from bert_score import score
import clip
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import logging

# Suppress BERTScore/transformers warnings about uninitialized pooler weights
warnings.filterwarnings('ignore', message='Some weights of RobertaModel were not initialized')
logging.getLogger("transformers").setLevel(logging.ERROR)

def calculate_text_similarity(text1, text2, sbert_model):
    """Enhanced text similarity calculation"""
    try:
        # SentenceBERT
        embeddings = sbert_model.encode([text1, text2])
        sbert_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # BERTScore (suppress model loading warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Some weights of RobertaModel')
            P, R, F1 = score([text2], [text1], lang="en", verbose=False)
        bertscore_f1 = F1.item()
        
        # Additional metrics
        # Word overlap (simple but effective)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
        
        return {
            'sbert_similarity': float(sbert_similarity),
            'bertscore_f1': float(bertscore_f1),
            'word_overlap': float(word_overlap)
        }
        
    except Exception as e:
        print(f"Error calculating text similarity: {e}")
        return {'sbert_similarity': 0.0, 'bertscore_f1': 0.0, 'word_overlap': 0.0}

def calculate_image_similarity(image1, image2, clip_model, clip_preprocess, device):
    """Enhanced image similarity calculation"""
    try:
        # Preprocess images
        img1_processed = clip_preprocess(image1).unsqueeze(0).to(device)
        img2_processed = clip_preprocess(image2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # CLIP features
            features1 = clip_model.encode_image(img1_processed)
            features2 = clip_model.encode_image(img2_processed)
            
            # Normalize features
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = torch.cosine_similarity(features1, features2).item()
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating image similarity: {e}")
        return 0.0

def bootstrap_text_similarity(text1, text2, sbert_model, n_bootstrap=1000):
    """Bootstrap text similarity calculation with confidence intervals"""
    try:
        # Get base similarity
        base_result = calculate_text_similarity(text1, text2, sbert_model)
        
        # Prepare for bootstrap
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        bootstrap_results = {
            'sbert_similarity': [],
            'bertscore_f1': [],
            'word_overlap': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling of words
            if len(words1) > 1:
                sampled_words1 = np.random.choice(words1, size=len(words1), replace=True)
                bootstrap_text1 = ' '.join(sampled_words1)
            else:
                bootstrap_text1 = text1
                
            if len(words2) > 1:
                sampled_words2 = np.random.choice(words2, size=len(words2), replace=True)
                bootstrap_text2 = ' '.join(sampled_words2)
            else:
                bootstrap_text2 = text2
            
            # Calculate similarity for bootstrap sample
            bootstrap_result = calculate_text_similarity(bootstrap_text1, bootstrap_text2, sbert_model)
            
            for metric in bootstrap_results:
                bootstrap_results[metric].append(bootstrap_result[metric])
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in bootstrap_results:
            values = np.array(bootstrap_results[metric])
            confidence_intervals[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
        
        return {
            'base_metrics': base_result,
            'bootstrap_ci': confidence_intervals,
            'n_bootstrap': n_bootstrap
        }
        
    except Exception as e:
        print(f"Error in bootstrap text similarity: {e}")
        return {
            'base_metrics': calculate_text_similarity(text1, text2, sbert_model),
            'bootstrap_ci': None,
            'n_bootstrap': 0
        }

def bootstrap_image_similarity(image1, image2, clip_model, clip_preprocess, device, n_bootstrap=1000):
    """Bootstrap image similarity calculation with confidence intervals"""
    try:
        # Get base similarity
        base_similarity = calculate_image_similarity(image1, image2, clip_model, clip_preprocess, device)
        
        # Convert images to tensors for bootstrap sampling
        img1_tensor = clip_preprocess(image1).unsqueeze(0).to(device)
        img2_tensor = clip_preprocess(image2).unsqueeze(0).to(device)
        
        bootstrap_similarities = []
        
        for _ in range(n_bootstrap):
            with torch.no_grad():
                # Add small random noise for bootstrap variation
                noise1 = torch.randn_like(img1_tensor) * 0.01
                noise2 = torch.randn_like(img2_tensor) * 0.01
                
                noisy_img1 = img1_tensor + noise1
                noisy_img2 = img2_tensor + noise2
                
                # Calculate features
                features1 = clip_model.encode_image(noisy_img1)
                features2 = clip_model.encode_image(noisy_img2)
                
                # Normalize features
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(features1, features2).item()
                bootstrap_similarities.append(similarity)
        
        # Calculate confidence intervals
        bootstrap_similarities = np.array(bootstrap_similarities)
        confidence_interval = {
            'mean': np.mean(bootstrap_similarities),
            'std': np.std(bootstrap_similarities),
            'ci_lower': np.percentile(bootstrap_similarities, 2.5),
            'ci_upper': np.percentile(bootstrap_similarities, 97.5)
        }
        
        return {
            'base_similarity': base_similarity,
            'bootstrap_ci': confidence_interval,
            'n_bootstrap': n_bootstrap
        }
        
    except Exception as e:
        print(f"Error in bootstrap image similarity: {e}")
        return {
            'base_similarity': calculate_image_similarity(image1, image2, clip_model, clip_preprocess, device),
            'bootstrap_ci': None,
            'n_bootstrap': 0
        }
