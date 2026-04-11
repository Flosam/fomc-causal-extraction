"""
Error Passage Analysis Script

Analyzes the 5 passages with validation errors in extractions_newest.csv
to identify root causes and patterns.
"""

import pandas as pd
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# File paths
EXTRACTIONS_FILE = Path("outputs/extractions_newest.csv")
PASSAGES_FILE = Path("outputs/passages.csv")
OUTPUT_DIR = Path("outputs")

def analyze_truncation_patterns(error_df: pd.DataFrame) -> Dict:
    """Analyze where and how LLM responses were truncated."""
    
    results = {
        'passages': [],
        'patterns': {
            'truncated_mid_field': 0,
            'truncated_mid_value': 0,
            'truncated_complete_structure': 0,
            'truncation_locations': []
        }
    }
    
    for _, row in error_df.iterrows():
        passage_id = row['passage_id']
        raw_response = str(row['raw_response'])
        error_msg = row['extraction_error']
        
        # Extract line/column info from error message
        line_match = re.search(r'line (\d+) column (\d+)', error_msg)
        error_line = int(line_match.group(1)) if line_match else None
        error_col = int(line_match.group(2)) if line_match else None
        
        # Determine truncation type
        last_50 = raw_response[-50:]
        truncation_type = 'unknown'
        
        if raw_response.endswith('"'):
            # Might be complete JSON that's malformed
            truncation_type = 'complete_but_malformed'
        elif '"effect":' in last_50 and not last_50.strip().endswith('}'):
            # Truncated mid-field
            truncation_type = 'mid_field'
            results['patterns']['truncated_mid_field'] += 1
        elif last_50.endswith('",') or last_50.endswith('"'):
            # Truncated mid-value or mid-string
            truncation_type = 'mid_value'
            results['patterns']['truncated_mid_value'] += 1
        else:
            # Other truncation
            truncation_type = 'other'
        
        # Count complete triplets before truncation
        complete_triplets = raw_response.count('"direction"')
        
        # Estimate total intended triplets
        cause_count = raw_response.count('"cause"')
        
        passage_info = {
            'passage_id': passage_id,
            'response_length': len(raw_response),
            'line_count': raw_response.count('\n'),
            'truncation_type': truncation_type,
            'error_at_line': error_line,
            'error_at_column': error_col,
            'last_50_chars': last_50,
            'complete_triplets': complete_triplets,
            'estimated_total_triplets': cause_count,
            'truncated_triplets': cause_count - complete_triplets
        }
        
        results['passages'].append(passage_info)
        results['patterns']['truncation_locations'].append({
            'passage_id': passage_id,
            'line': error_line,
            'char_position': len(raw_response)
        })
    
    return results


def identify_passage_characteristics(passages_df: pd.DataFrame, 
                                     error_ids: List[str],
                                     valid_ids: List[str]) -> Dict:
    """Extract and compare characteristics of error vs valid passages."""
    
    error_passages = passages_df[passages_df['passage_id'].isin(error_ids)]
    valid_passages = passages_df[passages_df['passage_id'].isin(valid_ids)]
    
    def compute_metrics(df: pd.DataFrame) -> Dict:
        texts = df['text']
        return {
            'count': len(df),
            'text_length': {
                'mean': texts.str.len().mean(),
                'std': texts.str.len().std(),
                'min': texts.str.len().min(),
                'max': texts.str.len().max(),
                'median': texts.str.len().median()
            },
            'word_count': {
                'mean': texts.str.split().str.len().mean(),
                'median': texts.str.split().str.len().median()
            },
            'sentence_count': {
                'mean': texts.str.count(r'[.!?]').mean(),
                'median': texts.str.count(r'[.!?]').median()
            },
            'special_chars': {
                'quotes_mean': texts.str.count('"').mean(),
                'parens_mean': texts.str.count(r'[()]').mean(),
                'commas_mean': texts.str.count(',').mean()
            },
            'estimated_tokens': {
                'mean': (texts.str.len() / 4).mean(),  # Rough estimate
                'median': (texts.str.len() / 4).median()
            }
        }
    
    return {
        'error_passages': compute_metrics(error_passages),
        'valid_passages': compute_metrics(valid_passages)
    }


def analyze_token_budget(extractions_df: pd.DataFrame, error_ids: List[str]) -> Dict:
    """Analyze token usage patterns for error vs valid passages."""
    
    error_rows = extractions_df[extractions_df['passage_id'].isin(error_ids)]
    valid_rows = extractions_df[~extractions_df['passage_id'].isin(error_ids)]
    
    # Group by passage to get per-passage totals
    def get_passage_tokens(df):
        # For each passage, sum all completion tokens (if multiple rows)
        # and get the prompt tokens (should be same for all rows from same passage)
        grouped = df.groupby('passage_id').agg({
            'prompt_tokens': 'first',
            'completion_tokens': 'sum'
        })
        grouped['total_tokens'] = grouped['prompt_tokens'] + grouped['completion_tokens']
        return grouped
    
    error_tokens = get_passage_tokens(error_rows)
    valid_tokens = get_passage_tokens(valid_rows)
    
    return {
        'error_passages': {
            'count': len(error_tokens),
            'prompt_tokens': {
                'mean': error_tokens['prompt_tokens'].mean(),
                'max': error_tokens['prompt_tokens'].max(),
                'min': error_tokens['prompt_tokens'].min()
            },
            'completion_tokens': {
                'mean': error_tokens['completion_tokens'].mean(),
                'max': error_tokens['completion_tokens'].max(),
                'min': error_tokens['completion_tokens'].min()
            },
            'total_tokens': {
                'mean': error_tokens['total_tokens'].mean(),
                'max': error_tokens['total_tokens'].max()
            }
        },
        'valid_passages': {
            'count': len(valid_tokens),
            'prompt_tokens': {
                'mean': valid_tokens['prompt_tokens'].mean(),
                'max': valid_tokens['prompt_tokens'].max(),
                'min': valid_tokens['prompt_tokens'].min()
            },
            'completion_tokens': {
                'mean': valid_tokens['completion_tokens'].mean(),
                'max': valid_tokens['completion_tokens'].max(),
                'min': valid_tokens['completion_tokens'].min()
            },
            'total_tokens': {
                'mean': valid_tokens['total_tokens'].mean(),
                'max': valid_tokens['total_tokens'].max()
            }
        },
        'raw_data': {
            'error_tokens': error_tokens.to_dict('index'),
            'valid_sample': valid_tokens.nlargest(5, 'total_tokens').to_dict('index')
        }
    }


def analyze_triplet_counts(extractions_df: pd.DataFrame, error_ids: List[str]) -> Dict:
    """Analyze correlation between triplet counts and failures."""
    
    # Get error passages and their estimated triplet counts
    error_rows = extractions_df[extractions_df['passage_id'].isin(error_ids)]
    error_triplets = {}
    
    for pid in error_ids:
        row = error_rows[error_rows['passage_id'] == pid].iloc[0]
        raw_resp = str(row['raw_response'])
        # Estimate by counting "cause" fields
        estimated = raw_resp.count('"cause"')
        error_triplets[pid] = estimated
    
    # Get valid passages and their actual triplet counts
    valid_rows = extractions_df[~extractions_df['passage_id'].isin(error_ids)]
    valid_rows = valid_rows[valid_rows['extraction_error'].isna()]  # Only truly valid
    
    valid_triplet_counts = valid_rows.groupby('passage_id').size()
    
    # Find valid passages with high triplet counts (6+)
    high_triplet_valid = valid_triplet_counts[valid_triplet_counts >= 6]
    
    return {
        'error_passages': {
            'triplet_counts': error_triplets,
            'mean': sum(error_triplets.values()) / len(error_triplets),
            'max': max(error_triplets.values()),
            'min': min(error_triplets.values())
        },
        'valid_passages': {
            'mean': valid_triplet_counts.mean(),
            'median': valid_triplet_counts.median(),
            'max': valid_triplet_counts.max(),
            'min': valid_triplet_counts.min(),
            'std': valid_triplet_counts.std(),
            'passages_with_6plus': len(high_triplet_valid),
            'high_triplet_examples': high_triplet_valid.head(10).to_dict()
        },
        'comparison': {
            'error_mean_vs_valid_mean': sum(error_triplets.values()) / len(error_triplets) / valid_triplet_counts.mean(),
            'errors_all_above_median': all(v > valid_triplet_counts.median() for v in error_triplets.values())
        }
    }


def find_similar_valid_passages(passages_df: pd.DataFrame, 
                                extractions_df: pd.DataFrame,
                                error_ids: List[str]) -> Dict:
    """Find valid passages with similar characteristics to error passages."""
    
    error_passages = passages_df[passages_df['passage_id'].isin(error_ids)]
    valid_passages = passages_df[~passages_df['passage_id'].isin(error_ids)]
    
    # Get valid passage IDs (that didn't have extraction errors)
    valid_extractions = extractions_df[extractions_df['extraction_error'].isna()]
    valid_passage_ids = valid_extractions['passage_id'].unique()
    valid_passages = valid_passages[valid_passages['passage_id'].isin(valid_passage_ids)]
    
    results = {}
    
    for _, error_passage in error_passages.iterrows():
        error_id = error_passage['passage_id']
        error_len = len(error_passage['text'])
        
        # Find valid passages within ±10% length
        lower_bound = error_len * 0.9
        upper_bound = error_len * 1.1
        
        similar = valid_passages[
            (valid_passages['text'].str.len() >= lower_bound) &
            (valid_passages['text'].str.len() <= upper_bound)
        ]
        
        if len(similar) > 0:
            results[error_id] = {
                'error_length': error_len,
                'similar_count': len(similar),
                'similar_passage_ids': similar['passage_id'].tolist()[:5],  # Top 5
                'similar_lengths': similar['text'].str.len().tolist()[:5]
            }
        else:
            # Expand search to ±20%
            lower_bound = error_len * 0.8
            upper_bound = error_len * 1.2
            similar = valid_passages[
                (valid_passages['text'].str.len() >= lower_bound) &
                (valid_passages['text'].str.len() <= upper_bound)
            ]
            results[error_id] = {
                'error_length': error_len,
                'similar_count': len(similar),
                'similar_passage_ids': similar['passage_id'].tolist()[:5] if len(similar) > 0 else [],
                'similar_lengths': similar['text'].str.len().tolist()[:5] if len(similar) > 0 else [],
                'note': 'expanded_to_20pct'
            }
    
    return results


def main():
    """Run all analyses and generate diagnostic outputs."""
    
    print("=" * 80)
    print("ERROR PASSAGE ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    extractions = pd.read_csv(EXTRACTIONS_FILE)
    passages = pd.read_csv(PASSAGES_FILE)
    
    # Identify error passages
    error_df = extractions[extractions['extraction_error'].notna()]
    error_ids = error_df['passage_id'].unique().tolist()
    valid_ids = extractions[extractions['extraction_error'].isna()]['passage_id'].unique().tolist()
    
    print(f"Error passages: {len(error_ids)}")
    print(f"Valid passages: {len(valid_ids)}")
    print(f"Error IDs: {error_ids}\n")
    
    # Run analyses
    analyses = {}
    
    print("1. Analyzing truncation patterns...")
    analyses['truncation'] = analyze_truncation_patterns(error_df)
    
    print("2. Identifying passage characteristics...")
    analyses['characteristics'] = identify_passage_characteristics(passages, error_ids, valid_ids)
    
    print("3. Analyzing token budgets...")
    analyses['tokens'] = analyze_token_budget(extractions, error_ids)
    
    print("4. Analyzing triplet counts...")
    analyses['triplets'] = analyze_triplet_counts(extractions, error_ids)
    
    print("5. Finding similar valid passages...")
    analyses['similar'] = find_similar_valid_passages(passages, extractions, error_ids)
    
    print("\nAnalysis complete! Saving results...")
    
    # Save results as JSON for inspection
    results_file = OUTPUT_DIR / "error_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(analyses), f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return analyses


if __name__ == "__main__":
    results = main()
