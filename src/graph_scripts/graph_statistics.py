"""
Generate summary statistics for FOMC knowledge graphs.

Computes network metrics and exports comparative statistics across periods.
"""

import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List

from src.config import ROOT


def load_graph(pkl_path: Path) -> nx.DiGraph:
    """Load NetworkX graph from pickle file."""
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    return G


def compute_graph_statistics(G: nx.DiGraph, period: str) -> Dict:
    """
    Compute comprehensive statistics for a knowledge graph.
    
    Args:
        G: NetworkX directed graph
        period: Period name
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'period': period,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    
    # Average degree
    degrees = dict(G.degree())
    stats['avg_degree'] = sum(degrees.values()) / len(degrees) if degrees else 0
    stats['max_degree'] = max(degrees.values()) if degrees else 0
    
    # Average edge weight
    weights = [data.get('weight', 1.0) for _, _, data in G.edges(data=True)]
    stats['avg_edge_weight'] = sum(weights) / len(weights) if weights else 0
    stats['max_edge_weight'] = max(weights) if weights else 0
    
    # Connectivity
    stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
    stats['strongly_connected_components'] = nx.number_strongly_connected_components(G)
    
    # Largest connected component
    if G.number_of_nodes() > 0:
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        stats['largest_component_nodes'] = len(largest_wcc)
        stats['largest_component_pct'] = len(largest_wcc) / G.number_of_nodes() * 100
    else:
        stats['largest_component_nodes'] = 0
        stats['largest_component_pct'] = 0
    
    # Self-loops (concept → itself)
    stats['self_loops'] = nx.number_of_selfloops(G)
    
    return stats


def get_top_nodes(G: nx.DiGraph, n: int = 10) -> pd.DataFrame:
    """Get top n nodes by degree centrality."""
    centrality = nx.degree_centrality(G)
    degrees = dict(G.degree())
    
    data = []
    for node in G.nodes():
        data.append({
            'node': node,
            'degree': degrees[node],
            'degree_centrality': centrality[node],
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('degree', ascending=False).head(n)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed
    
    return df


def get_top_edges(G: nx.DiGraph, n: int = 10) -> pd.DataFrame:
    """Get top n edges by frequency."""
    data = []
    for u, v, edge_data in G.edges(data=True):
        data.append({
            'cause': u,
            'effect': v,
            'frequency': edge_data.get('frequency', 1),
            'direction': edge_data.get('direction', ''),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('frequency', ascending=False).head(n)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed
    
    return df


def main():
    """Generate statistics for all three period graphs."""
    # Setup paths
    graph_dir = ROOT / 'outputs' / 'knowledge_graphs'
    output_dir = ROOT / 'outputs'
    
    # Period configurations
    periods = {
        'great_moderation': 'Great Moderation',
        'post_crisis_zlb': 'Post-Crisis ZLB',
        'post_covid': 'Post-COVID',
    }
    
    print("=" * 70)
    print("COMPUTING KNOWLEDGE GRAPH STATISTICS")
    print("=" * 70)
    
    # Collect overall statistics
    all_stats = []
    
    for period_key, period_name in periods.items():
        print(f"\n{period_name}:")
        print("-" * 70)
        
        # Load graph
        graph_path = graph_dir / f'graph_{period_key}.pkl'
        G = load_graph(graph_path)
        
        # Compute statistics
        stats = compute_graph_statistics(G, period_name)
        all_stats.append(stats)
        
        # Print summary
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Edges: {stats['edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Avg Edge Weight: {stats['avg_edge_weight']:.2f}")
        print(f"  Self-loops: {stats['self_loops']}")
        print(f"  Largest Component: {stats['largest_component_nodes']} nodes ({stats['largest_component_pct']:.1f}%)")
        
        # Get top nodes
        print(f"\n  Top 10 Nodes by Degree:")
        top_nodes = get_top_nodes(G, 10)
        for idx, row in top_nodes.iterrows():
            print(f"    {idx:2}. {row['node']:30} (deg={row['degree']:2}, in={row['in_degree']:2}, out={row['out_degree']:2})")
        
        # Get top edges
        print(f"\n  Top 10 Edges by Frequency:")
        top_edges = get_top_edges(G, 10)
        for idx, row in top_edges.iterrows():
            direction_str = f"[{row['direction']}]" if row['direction'] else ""
            print(f"    {idx:2}. {row['cause']:25} → {row['effect']:25} (freq={row['frequency']}) {direction_str}")
        
        # Save detailed stats per period
        top_nodes.to_csv(output_dir / f'kg_top_nodes_{period_key}.csv', index=True)
        top_edges.to_csv(output_dir / f'kg_top_edges_{period_key}.csv', index=True)
        print(f"\n  Saved detailed stats to kg_top_nodes_{period_key}.csv and kg_top_edges_{period_key}.csv")
    
    # Save overall comparison table
    print(f"\n{'=' * 70}")
    print("CROSS-PERIOD COMPARISON")
    print("=" * 70)
    
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.round(4)
    
    print("\n" + stats_df.to_string(index=False))
    
    stats_path = output_dir / 'graph_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved comparison table to {stats_path}")
    
    print(f"\n{'=' * 70}")
    print("✓ Successfully generated all statistics")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
