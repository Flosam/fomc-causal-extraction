"""
Build NetworkX knowledge graphs from FOMC causal triples.

Creates one directed graph per economic period with standardized nodes
and weighted edges based on relationship frequency.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
from collections import defaultdict

from src.standardize_terms import standardize_term
from src.config import ROOT


def load_annotated_data(csv_path: Path) -> pd.DataFrame:
    """Load and filter annotated triples to faithful ones plus failure mode 4."""
    df = pd.read_csv(csv_path)
    
    # Filter to faithful triples OR failure mode 4 (which we now count as valid)
    valid = df[(df['human_faithful'] == 1) | (df['human_failure_mode'] == 4)].copy()
    
    print(f"Loaded {len(df)} total triples")
    print(f"Filtered to {len(valid)} valid triples ({len(valid)/len(df)*100:.1f}%)")
    print(f"  - Faithful: {(df['human_faithful'] == 1).sum()}")
    print(f"  - Failure mode 4: {(df['human_failure_mode'] == 4).sum()}")
    
    # Show breakdown by period
    print("\nValid triples by period:")
    for period, count in valid['period'].value_counts().sort_index().items():
        print(f"  {period}: {count}")
    
    return valid


def build_period_graph(
    period_df: pd.DataFrame, 
    period_name: str
) -> nx.DiGraph:
    """
    Build a weighted directed graph for one economic period.
    
    Args:
        period_df: DataFrame filtered to one period's faithful triples
        period_name: Name of the period for logging
        
    Returns:
        NetworkX DiGraph with weighted edges
    """
    G = nx.DiGraph()
    
    # Track edge information for aggregation
    # Key: (cause, effect), Value: list of instance data
    edge_data = defaultdict(list)
    
    standardization_stats = defaultdict(set)  # Track how many raw terms map to each concept
    dropped_count = 0  # Track dropped non-economic triples
    
    for _, row in period_df.iterrows():
        cause_raw = row['cause']
        effect_raw = row['effect']
        triple_id = row['triple_id']
        
        # Standardize terms (drop unmapped non-economic terms)
        cause_std = standardize_term(cause_raw, drop_unmapped=True)
        effect_std = standardize_term(effect_raw, drop_unmapped=True)
        
        # Skip empty standardizations (non-economic or unmapped terms)
        if not cause_std or not effect_std:
            dropped_count += 1
            continue
        
        # Track standardization coverage
        standardization_stats[cause_std].add(cause_raw)
        standardization_stats[effect_std].add(effect_raw)
        
        # Store edge data
        edge_data[(cause_std, effect_std)].append({
            'triple_id': triple_id,
            'original_cause': cause_raw,
            'original_effect': effect_raw,
            'direction': row.get('direction', ''),
        })
    
    # Build graph edges with aggregated weights
    for (cause, effect), instances in edge_data.items():
        frequency = len(instances)
        # Weight is simply the frequency count
        weight = frequency
        
        # Collect example triples (up to 3)
        examples = [
            f"{inst['original_cause']} → {inst['original_effect']}"
            for inst in instances[:3]
        ]
        
        # Count direction types
        directions = [inst['direction'] for inst in instances if inst['direction']]
        direction_counts = pd.Series(directions).value_counts().to_dict() if directions else {}
        
        # Determine primary direction (most common)
        primary_direction = max(direction_counts, key=direction_counts.get) if direction_counts else ''
        
        # Add edge with attributes
        G.add_edge(
            cause,
            effect,
            weight=weight,
            frequency=frequency,
            examples=examples,
            direction=primary_direction,
            direction_counts=direction_counts,
            triple_ids=[inst['triple_id'] for inst in instances],
        )
    
    print(f"\n{period_name} graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    print(f"  Standardized concepts: {len(standardization_stats)}")
    print(f"  Avg raw terms per concept: {sum(len(v) for v in standardization_stats.values()) / len(standardization_stats):.1f}")
    print(f"  Dropped non-economic triples: {dropped_count}")
    
    return G


def get_top_nodes(G: nx.DiGraph, n: int = 10) -> List[Tuple[str, float]]:
    """Get top n nodes by degree centrality."""
    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:n]


def get_top_edges(G: nx.DiGraph, n: int = 10) -> List[Tuple[str, str, float]]:
    """Get top n edges by weight."""
    edges_with_weight = [
        (u, v, data['weight'])
        for u, v, data in G.edges(data=True)
    ]
    sorted_edges = sorted(edges_with_weight, key=lambda x: x[2], reverse=True)
    return sorted_edges[:n]


def save_graph(G: nx.DiGraph, path: Path, period_name: str):
    """Save graph to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved {period_name} graph to {path}")


def main():
    """Build knowledge graphs for all three periods."""
    # Setup paths
    annotated_csv = ROOT / 'outputs' / 'annotated.csv'
    output_dir = ROOT / 'outputs' / 'knowledge_graphs'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("=" * 70)
    print("BUILDING FOMC KNOWLEDGE GRAPHS")
    print("=" * 70)
    df = load_annotated_data(annotated_csv)
    
    # Build graphs per period
    graphs = {}
    periods = ['great_moderation', 'post_crisis_zlb', 'post_covid']
    
    for period in periods:
        print(f"\n{'=' * 70}")
        print(f"Building graph for: {period.upper().replace('_', ' ')}")
        print(f"{'=' * 70}")
        
        period_df = df[df['period'] == period]
        G = build_period_graph(period_df, period)
        graphs[period] = G
        
        # Show top nodes and edges
        print(f"\nTop 10 nodes by degree centrality:")
        for i, (node, centrality) in enumerate(get_top_nodes(G, 10), 1):
            degree = G.degree(node)
            print(f"  {i:2}. {node:30} (degree={degree}, centrality={centrality:.3f})")
        
        print(f"\nTop 10 edges by weight:")
        for i, (cause, effect, weight) in enumerate(get_top_edges(G, 10), 1):
            freq = G[cause][effect]['frequency']
            print(f"  {i:2}. {cause:25} → {effect:25} (weight={weight:.2f}, freq={freq})")
        
        # Save graph
        graph_path = output_dir / f'graph_{period}.pkl'
        save_graph(G, graph_path, period)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Successfully built {len(graphs)} knowledge graphs")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 70}")
    
    return graphs


if __name__ == "__main__":
    main()
