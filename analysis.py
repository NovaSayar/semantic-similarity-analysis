# Analysis and visualization functions

import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
import hashlib
import warnings
import logging
from data_loader import load_prompts_from_csv
from similarity_metrics import calculate_text_similarity, calculate_image_similarity, bootstrap_text_similarity, bootstrap_image_similarity

# Suppress transformers warnings
warnings.filterwarnings('ignore', message='Some weights of RobertaModel were not initialized')
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_full_analysis(pipeline, csv_file_path):
    """Run comprehensive analysis with all prompts from CSV"""
    results = []
    
    print("Starting Full Semantic Similarity Analysis from CSV")
    
    # Load prompts from CSV
    prompts_dict = load_prompts_from_csv(csv_file_path)
    
    # Process all prompts
    total_prompts = sum(len(prompts) for category_prompts in prompts_dict.values() 
                        for prompts in category_prompts.values())
    print(f"Total prompts to process: {total_prompts}")
    
    current_prompt = 0
    
    # Test all combinations
    for category, prompts_by_type in prompts_dict.items():
        for prompt_type, prompts in prompts_by_type.items():
            for i, prompt in enumerate(prompts):
                current_prompt += 1
                prompt_id = f"{category[:1].upper()}{prompt_type[:1].upper()}_{i+1:02d}"
                
                # Check if already processed
                pattern = f"images/**/result_{category}_{prompt_type}_{prompt_id}_seed*.json"
                if glob.glob(pattern, recursive=True):
                    print(f"Skipping already processed prompt: {prompt_id}")
                    continue

                print(f"\nProgress: {current_prompt}/{total_prompts} → Running: {prompt_id}")
                
                # Generate reproducible seed
                seed = int(hashlib.md5(f"{category}_{prompt_type}_{i}_{prompt[:20]}".encode()).hexdigest(), 16) % 10000

                try:
                    result = complete_analysis_pipeline(
                        pipeline, prompt, category, prompt_type, prompt_id, seed
                    )
                    if result:
                        results.append(result)
                    else:
                        print(f"Skipped prompt (returned None): {prompt_id}")
                except Exception as e:
                    print(f"Error in prompt {prompt_id}: {e}")
    
    print(f"\nCompleted processing {len(results)} prompts successfully")
    return analyze_results(results)

def complete_analysis_pipeline(pipeline, original_prompt, category, prompt_type, prompt_id, seed=42, save_images=True, n_bootstrap=1000):
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
    
    # Run pipeline safely
    try:
        pipeline_result = pipeline.complete_pipeline(original_prompt, category, prompt_type, prompt_id, seed, save_images)
        if pipeline_result is None:
            print(f"Skipped prompt (pipeline returned None): {prompt_id}")
            return None
    except Exception as e:
        print(f"Error during pipeline execution for {prompt_id}: {e}")
        return None

    
    image1 = pipeline_result['image1']
    image2 = pipeline_result['image2']
    caption2 = pipeline_result['caption2']
    
    # Stage 4: Semantic Analysis
    print("Stage 4: Semantic Analysis")
    
    # Text similarity (C1 vs C2) with Bootstrap Confidence Intervals
    print("Computing text similarity with bootstrap confidence intervals...")
    text_bootstrap = bootstrap_text_similarity(original_prompt, caption2, pipeline.sbert_model, n_bootstrap)
    text_metrics = text_bootstrap['base_metrics']
    text_ci = text_bootstrap['bootstrap_ci']
    
    # Image similarity (I1 vs I2) with Bootstrap Confidence Intervals  
    print("Computing image similarity with bootstrap confidence intervals...")
    image_bootstrap = bootstrap_image_similarity(image1, image2, pipeline.clip_model, pipeline.clip_preprocess, pipeline.device, n_bootstrap)
    image_similarity = image_bootstrap['base_similarity']
    image_ci = image_bootstrap['bootstrap_ci']
    
    # Results
    result = {
        'timestamp': datetime.now().isoformat(),
        'prompt_id': prompt_id,
        'category': category,
        'prompt_type': prompt_type,
        'original_prompt': original_prompt,
        'generated_caption': caption2,
        'seed': seed,
        'sbert_similarity': text_metrics['sbert_similarity'],
        'bertscore_f1': text_metrics['bertscore_f1'],
        'word_overlap': text_metrics['word_overlap'],
        'clip_image_similarity': image_similarity,
        # Bootstrap confidence intervals
        'sbert_ci_lower': text_ci['sbert_similarity']['ci_lower'] if text_ci else None,
        'sbert_ci_upper': text_ci['sbert_similarity']['ci_upper'] if text_ci else None,
        'bertscore_ci_lower': text_ci['bertscore_f1']['ci_lower'] if text_ci else None,
        'bertscore_ci_upper': text_ci['bertscore_f1']['ci_upper'] if text_ci else None,
        'word_overlap_ci_lower': text_ci['word_overlap']['ci_lower'] if text_ci else None,
        'word_overlap_ci_upper': text_ci['word_overlap']['ci_upper'] if text_ci else None,
        'clip_ci_lower': image_ci['ci_lower'] if image_ci else None,
        'clip_ci_upper': image_ci['ci_upper'] if image_ci else None,
        'bootstrap_samples': text_bootstrap['n_bootstrap'],
        'avg_text_similarity': (text_metrics['sbert_similarity'] + text_metrics['bertscore_f1']) / 2,
        'overall_similarity': (text_metrics['sbert_similarity'] + text_metrics['bertscore_f1'] + image_similarity) / 3
    }
    
    # Print metrics with Bootstrap Confidence Intervals
    print(f"\nMetrics with 95% Bootstrap Confidence Intervals:")
    if text_ci:
        print(f"  SentenceBERT Similarity: {text_metrics['sbert_similarity']:.3f} "
              f"(95% CI: {text_ci['sbert_similarity']['ci_lower']:.3f}-{text_ci['sbert_similarity']['ci_upper']:.3f})")
        print(f"  BERTScore F1: {text_metrics['bertscore_f1']:.3f} "
              f"(95% CI: {text_ci['bertscore_f1']['ci_lower']:.3f}-{text_ci['bertscore_f1']['ci_upper']:.3f})")
        print(f"  Word Overlap: {text_metrics['word_overlap']:.3f} "
              f"(95% CI: {text_ci['word_overlap']['ci_lower']:.3f}-{text_ci['word_overlap']['ci_upper']:.3f})")
    else:
        print(f"  SentenceBERT Similarity: {text_metrics['sbert_similarity']:.3f}")
        print(f"  BERTScore F1: {text_metrics['bertscore_f1']:.3f}")
        print(f"  Word Overlap: {text_metrics['word_overlap']:.3f}")
    
    if image_ci:
        print(f"  CLIP Image Similarity: {image_similarity:.3f} "
              f"(95% CI: {image_ci['ci_lower']:.3f}-{image_ci['ci_upper']:.3f})")
    else:
        print(f"  CLIP Image Similarity: {image_similarity:.3f}")
    
    print(f"  Overall Similarity: {result['overall_similarity']:.3f}")
    print(f"  Bootstrap Samples: {text_bootstrap['n_bootstrap']}")
    
    # Save individual result
    if save_images:
        with open(f"{results_dir}/result_{category}_{prompt_type}_{prompt_id}_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    return result

def analyze_results(results):
    """Enhanced results analysis"""
    if not results:
        print("❌ No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE SEMANTIC SIMILARITY ANALYSIS")
    print(f"{'='*80}")
    
    # Overall statistics
    print(f"\n📈 OVERALL STATISTICS (n={len(df)}):")
    metrics = ['sbert_similarity', 'bertscore_f1', 'word_overlap', 'clip_image_similarity', 'overall_similarity']
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"  {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Hypothesis testing
    test_hypotheses(df)
    
    # Detailed analysis
    detailed_analysis(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'semantic_similarity_results_{timestamp}.csv', index=False)
    print(f"\n💾 Results saved to 'semantic_similarity_results_{timestamp}.csv'")
    
    # Create comprehensive visualization
    create_visualizations(df, timestamp)
    
    return df

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (group1.mean() - group2.mean()) / pooled_std

def descriptive_statistics(df):
    """Calculate descriptive statistics for pilot study"""
    print(f"\nDESCRIPTIVE STATISTICS (n={len(df)}):")
    print(f"{'='*60}")
    
    metrics = ['sbert_similarity', 'bertscore_f1', 'word_overlap', 'clip_image_similarity', 'overall_similarity']
    
    for metric in metrics:
        data = df[metric]
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std Dev: {data.std():.4f}")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Median: {data.median():.4f}")
        print(f"  IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}")
        print(f"  Sample Size: {len(data)}")
        
        # Normality test (Shapiro-Wilk)
        if len(data) >= 3:  # Minimum sample size for Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(data)
            print(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
            print(f"  Normality: {'Normal' if shapiro_p > 0.05 else 'Non-normal'}")
        else:
            print(f"  Shapiro-Wilk test: Insufficient sample size")

def pilot_study_analysis(df):
    """Pilot study specific analysis with non-parametric tests"""
    print(f"\nPILOT STUDY STATISTICAL ANALYSIS:")
    print(f"{'='*60}")
    
    # Descriptive statistics
    descriptive_statistics(df)
    
    # H1: Content Type Hypothesis - Non-parametric test
    print(f"\nH1 - CONTENT TYPE HYPOTHESIS (Non-parametric):")
    place_scores = df[df['category'] == 'place']['overall_similarity']
    nonplace_scores = df[df['category'] == 'non_place']['overall_similarity']
    
    print(f"  Place prompts: n={len(place_scores)}, median={place_scores.median():.4f}")
    print(f"  Non-place prompts: n={len(nonplace_scores)}, median={nonplace_scores.median():.4f}")
    
    if len(place_scores) > 0 and len(nonplace_scores) > 0:
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(place_scores, nonplace_scores, alternative='two-sided')
        effect_size = calculate_cohens_d(place_scores, nonplace_scores)
        
        print(f"  Mann-Whitney U test: U={statistic:.2f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size interpretation: {effect_interpretation}")
        print(f"  Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Hypothesis: {'SUPPORTED' if place_scores.median() > nonplace_scores.median() else 'NOT SUPPORTED'}")
    
    # H2: Complexity Hypothesis - Non-parametric test
    print(f"\nH2 - COMPLEXITY HYPOTHESIS (Non-parametric):")
    vague_scores = df[df['prompt_type'] == 'vague']['overall_similarity']
    structured_scores = df[df['prompt_type'] == 'structured']['overall_similarity']
    
    print(f"  Vague prompts: n={len(vague_scores)}, median={vague_scores.median():.4f}")
    print(f"  Structured prompts: n={len(structured_scores)}, median={structured_scores.median():.4f}")
    
    if len(vague_scores) > 0 and len(structured_scores) > 0:
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(structured_scores, vague_scores, alternative='two-sided')
        effect_size = calculate_cohens_d(structured_scores, vague_scores)
        
        print(f"  Mann-Whitney U test: U={statistic:.2f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size interpretation: {effect_interpretation}")
        print(f"  Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Hypothesis: {'SUPPORTED' if structured_scores.median() > vague_scores.median() else 'NOT SUPPORTED'}")
    
    # H3: Multimodal metrics correlation - Spearman correlation (non-parametric)
    print(f"\nH3 - MULTIMODAL ADVANTAGE (Non-parametric correlation):")
    text_avg = (df['sbert_similarity'] + df['bertscore_f1']) / 2
    image_sim = df['clip_image_similarity']
    
    spearman_corr, spearman_p = stats.spearmanr(text_avg, image_sim)
    
    print(f"  Spearman correlation: r={spearman_corr:.4f}, p={spearman_p:.4f}")
    print(f"  Statistical significance: {'Yes' if spearman_p < 0.05 else 'No'}")
    print(f"  Hypothesis: {'SUPPORTED' if abs(spearman_corr) > 0.5 else 'NOT SUPPORTED'}")
    
    # Multiple groups analysis (if applicable)
    groups = df.groupby(['category', 'prompt_type'])['overall_similarity'].apply(list)
    if len(groups) > 2:
        print(f"\nMULTIPLE GROUPS ANALYSIS:")
        group_values = [scores for scores in groups.values() if len(scores) > 0]
        
        if len(group_values) > 2:
            # Kruskal-Wallis test for multiple groups
            kruskal_stat, kruskal_p = stats.kruskal(*group_values)
            print(f"  Kruskal-Wallis test: H={kruskal_stat:.4f}, p={kruskal_p:.4f}")
            print(f"  Statistical significance: {'Yes' if kruskal_p < 0.05 else 'No'}")
            print(f"  Interpretation: {'Significant differences between groups' if kruskal_p < 0.05 else 'No significant differences between groups'}")

def test_hypotheses(df):
    """Test research hypotheses with statistical analysis"""
    print(f"\n🔬 HYPOTHESIS TESTING:")
    
    # H1: Content Type Hypothesis
    place_scores = df[df['category'] == 'place']['overall_similarity']
    nonplace_scores = df[df['category'] == 'non_place']['overall_similarity']
    
    print(f"\nH1 - Content Type Hypothesis:")
    print(f"  Place prompts: {place_scores.mean():.3f} ± {place_scores.std():.3f} (n={len(place_scores)})")
    print(f"  Non-place prompts: {nonplace_scores.mean():.3f} ± {nonplace_scores.std():.3f} (n={len(nonplace_scores)})")
    print(f"  Difference: {place_scores.mean() - nonplace_scores.mean():.3f}")
    print(f"  Result: {'✅ SUPPORTED' if place_scores.mean() > nonplace_scores.mean() else '❌ NOT SUPPORTED'}")
    
    # H2: Complexity Hypothesis
    vague_scores = df[df['prompt_type'] == 'vague']['overall_similarity']
    structured_scores = df[df['prompt_type'] == 'structured']['overall_similarity']
    
    print(f"\nH2 - Complexity Hypothesis:")
    print(f"  Vague prompts: {vague_scores.mean():.3f} ± {vague_scores.std():.3f} (n={len(vague_scores)})")
    print(f"  Structured prompts: {structured_scores.mean():.3f} ± {structured_scores.std():.3f} (n={len(structured_scores)})")
    print(f"  Difference: {structured_scores.mean() - vague_scores.mean():.3f}")
    print(f"  Result: {'✅ SUPPORTED' if structured_scores.mean() > vague_scores.mean() else '❌ NOT SUPPORTED'}")
    
    # H3: Multimodal metrics correlation
    text_avg = (df['sbert_similarity'] + df['bertscore_f1']) / 2
    image_sim = df['clip_image_similarity']
    correlation = text_avg.corr(image_sim)
    
    print(f"\nH3 - Multimodal Advantage:")
    print(f"  Text-Image correlation: {correlation:.3f}")
    print(f"  Result: {'✅ SUPPORTED' if abs(correlation) > 0.5 else '❌ NOT SUPPORTED'}")

def detailed_analysis(df):
    """Detailed analysis of results"""
    print(f"\n📋 DETAILED ANALYSIS:")
    
    # Best and worst cases
    best_idx = df['overall_similarity'].idxmax()
    worst_idx = df['overall_similarity'].idxmin()
    
    print(f"\n🏆 BEST SEMANTIC PRESERVATION:")
    best = df.loc[best_idx]
    print(f"  ID: {best['prompt_id']}")
    print(f"  Category: {best['category']} - {best['prompt_type']}")
    print(f"  Overall similarity: {best['overall_similarity']:.3f}")
    print(f"  Original: {best['original_prompt'][:80]}...")
    print(f"  Caption: {best['generated_caption'][:80]}...")
    
    print(f"\n❌ WORST SEMANTIC PRESERVATION:")
    worst = df.loc[worst_idx]
    print(f"  ID: {worst['prompt_id']}")
    print(f"  Category: {worst['category']} - {worst['prompt_type']}")
    print(f"  Overall similarity: {worst['overall_similarity']:.3f}")
    print(f"  Original: {worst['original_prompt'][:80]}...")
    print(f"  Caption: {worst['generated_caption'][:80]}...")
    
    # Category-Type combinations
    print(f"\n🔄 CATEGORY-TYPE COMBINATIONS:")
    df['combination'] = df['category'] + '_' + df['prompt_type']
    combo_stats = df.groupby('combination')['overall_similarity'].agg(['mean', 'std', 'count'])
    print(combo_stats.round(3))

def create_visualizations(df, timestamp):
    """Create comprehensive visualizations"""
    # Create images folder if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Create the combined figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Overall metrics comparison
    metrics = ['sbert_similarity', 'bertscore_f1', 'word_overlap', 'clip_image_similarity']
    means = [df[metric].mean() for metric in metrics]
    stds = [df[metric].std() for metric in metrics]
    
    axes[0,0].bar(range(len(metrics)), means, yerr=stds, capsize=5)
    axes[0,0].set_xticks(range(len(metrics)))
    axes[0,0].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
    axes[0,0].set_title('Overall Metric Performance')
    axes[0,0].set_ylabel('Similarity Score')
    
    # Plot 2: Category comparison
    category_data = df.groupby('category')[metrics].mean()
    category_data.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Category Comparison')
    axes[0,1].set_ylabel('Mean Similarity')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Prompt type comparison
    type_data = df.groupby('prompt_type')[metrics].mean()
    type_data.plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Prompt Type Comparison')
    axes[0,2].set_ylabel('Mean Similarity')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Correlation matrix
    corr_matrix = df[metrics + ['overall_similarity']].corr()
    im = axes[1,0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,0].set_xticks(range(len(corr_matrix.columns)))
    axes[1,0].set_yticks(range(len(corr_matrix.columns)))
    axes[1,0].set_xticklabels([c.replace('_', '\n') for c in corr_matrix.columns], rotation=45)
    axes[1,0].set_yticklabels([c.replace('_', '\n') for c in corr_matrix.columns])
    axes[1,0].set_title('Metric Correlations')
    plt.colorbar(im, ax=axes[1,0])
    
    # Plot 5: Distribution of overall similarities
    axes[1,1].hist(df['overall_similarity'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(df['overall_similarity'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].set_xlabel('Overall Similarity')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Overall Similarities')
    axes[1,1].legend()
    
    # Plot 6: Scatter plot of text vs image similarity
    axes[1,2].scatter(df['avg_text_similarity'], df['clip_image_similarity'], 
                     c=df['overall_similarity'], cmap='viridis', alpha=0.7)
    axes[1,2].set_xlabel('Average Text Similarity')
    axes[1,2].set_ylabel('Image Similarity')
    axes[1,2].set_title('Text vs Image Similarity')
    
    plt.tight_layout()
    
    # Save combined figure
    plt.savefig(f'images/semantic_similarity_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Now create and save individual figures
    print("Creating individual figure files...")
    
    # Individual Plot 1: Overall metrics comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(len(metrics)), means, yerr=stds, capsize=5)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
    ax1.set_title('Overall Metric Performance')
    ax1.set_ylabel('Similarity Score')
    plt.tight_layout()
    plt.savefig(f'images/1_overall_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Plot 2: Category comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    category_data.plot(kind='bar', ax=ax2)
    ax2.set_title('Category Comparison')
    ax2.set_ylabel('Mean Similarity')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/2_category_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Plot 3: Prompt type comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    type_data.plot(kind='bar', ax=ax3)
    ax3.set_title('Prompt Type Comparison')
    ax3.set_ylabel('Mean Similarity')
    ax3.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/3_prompt_type_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Plot 4: Correlation matrix
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels([c.replace('_', '\n') for c in corr_matrix.columns], rotation=45)
    ax4.set_yticklabels([c.replace('_', '\n') for c in corr_matrix.columns])
    ax4.set_title('Metric Correlations')
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.savefig(f'images/4_correlation_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Plot 5: Distribution of overall similarities
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.hist(df['overall_similarity'], bins=15, alpha=0.7, edgecolor='black')
    ax5.axvline(df['overall_similarity'].mean(), color='red', linestyle='--', label='Mean')
    ax5.set_xlabel('Overall Similarity')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Overall Similarities')
    ax5.legend()
    plt.tight_layout()
    plt.savefig(f'images/5_similarity_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Plot 6: Scatter plot of text vs image similarity
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    scatter = ax6.scatter(df['avg_text_similarity'], df['clip_image_similarity'], 
                         c=df['overall_similarity'], cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Average Text Similarity')
    ax6.set_ylabel('Image Similarity')
    ax6.set_title('Text vs Image Similarity')
    plt.colorbar(scatter, ax=ax6)
    plt.tight_layout()
    plt.savefig(f'images/6_text_vs_image_scatter_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Combined visualization saved to 'images/semantic_similarity_analysis_{timestamp}.png'")
    print(f"📊 Individual figures saved as:")
    print(f"   - images/1_overall_metrics_{timestamp}.png")
    print(f"   - images/2_category_comparison_{timestamp}.png")
    print(f"   - images/3_prompt_type_comparison_{timestamp}.png")
    print(f"   - images/4_correlation_matrix_{timestamp}.png")
    print(f"   - images/5_similarity_distribution_{timestamp}.png")
    print(f"   - images/6_text_vs_image_scatter_{timestamp}.png")

def run_pilot_study_tests(df):
    """Perform non-parametric tests suitable for pilot study"""
    print("\nPILOT STUDY NON-PARAMETRIC ANALYSIS")
    print("="*60)
    
    try:
        # Run the comprehensive pilot study analysis
        pilot_study_analysis(df)
        
        print(f"\nPILOT STUDY RECOMMENDATIONS:")
        print(f"- Sample size: {len(df)} (suitable for pilot study)")
        print(f"- Non-parametric tests used (robust for small samples)")
        print(f"- Bootstrap confidence intervals available in similarity_metrics.py")
        print(f"- Effect sizes calculated for practical significance")
        print(f"- Results provide preliminary insights for main study design")
        
    except Exception as e:
        print(f"Error in pilot study analysis: {e}")

def run_two_way_anova(df):
    """Perform two-way ANOVA analysis"""
    print("\n📊 TWO-WAY ANOVA RESULTS")
    try:
        model = ols('overall_similarity ~ C(category) + C(prompt_type) + C(category):C(prompt_type)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        
        # Effect sizes
        print(f"\n📈 EFFECT SIZES:")
        eta_squared = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
        for factor, eta_sq in eta_squared.items():
            if factor != 'Residual':
                print(f"  {factor}: η² = {eta_sq:.3f}")
                
    except Exception as e:
        print(f"❌ Error in ANOVA analysis: {e}")
