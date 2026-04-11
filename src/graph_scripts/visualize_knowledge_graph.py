"""
Visualize FOMC knowledge graphs with NetworkX and matplotlib.

Creates publication-quality network visualizations showing causal relationships
between economic concepts across different time periods.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
from typing import Dict, Tuple
import numpy as np

from src.config import ROOT


# Color scheme for different node categories
NODE_COLORS = {
    "inflation": "#e74c3c",  # Red
    "economic_activity": "#3498db",  # Blue
    "monetary_policy": "#9b59b6",  # Purple
    "financial_conditions": "#f39c12",  # Orange
    "consumer_spending": "#2ecc71",  # Green
    "labor_market": "#1abc9c",  # Teal
    "housing": "#e67e22",  # Dark orange
    "business_investment": "#16a085",  # Dark teal
    "aggregate_demand": "#27ae60",  # Dark green
    "energy_prices": "#d35400",  # Dark orange-red
    "trade": "#34495e",  # Dark blue-gray
    "supply": "#8e44ad",  # Dark purple
    "immigration": "#95a5a6",  # Gray
    "expectations": "#c0392b",  # Dark red
    "default": "#7f8c8d",  # Default gray
}

# Color scheme for edge directions
EDGE_DIRECTION_COLORS = {
    "positive": "#27ae60",  # Green
    "negative": "#e74c3c",  # Red
    "ambiguous": "#f39c12",  # Orange
    "": "#95a5a6",  # Gray for empty/unknown
}


def load_graph(pkl_path: Path) -> nx.DiGraph:
    """Load NetworkX graph from pickle file."""
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    return G


def get_node_colors(G: nx.DiGraph) -> list:
    """Assign colors to nodes based on their concept category."""
    colors = []
    for node in G.nodes():
        colors.append(NODE_COLORS.get(node, NODE_COLORS["default"]))
    return colors


def get_node_sizes(G: nx.DiGraph, base_size: int = 300, scale: float = 3000) -> list:
    """
    Calculate node sizes based on degree centrality.
    
    Args:
        G: NetworkX graph
        base_size: Minimum node size
        scale: Scaling factor for centrality
        
    Returns:
        List of node sizes
    """
    centrality = nx.degree_centrality(G)
    sizes = [base_size + centrality[node] * scale for node in G.nodes()]
    return sizes


def get_edge_widths(G: nx.DiGraph, base_width: float = 1.0, scale: float = 3.5) -> list:
    """
    Calculate edge widths based on frequency.
    
    Args:
        G: NetworkX graph
        base_width: Minimum edge width
        scale: Scaling factor
        
    Returns:
        List of edge widths
    """
    frequencies = [data.get('frequency', 1) for _, _, data in G.edges(data=True)]
    max_freq = max(frequencies) if frequencies else 1
    
    # Linear scaling
    widths = [base_width + (freq / max_freq) * scale for freq in frequencies]
    return widths


def get_edge_colors(G: nx.DiGraph) -> list:
    """
    Get edge colors based on direction attribute.
    
    Returns:
        List of edge colors
    """
    colors = []
    for _, _, data in G.edges(data=True):
        direction = data.get('direction', '')
        colors.append(EDGE_DIRECTION_COLORS.get(direction, EDGE_DIRECTION_COLORS['']))
    return colors


def create_legend(ax, G: nx.DiGraph):
    """Add legends showing node categories and edge directions."""
    # Node legend - show all present categories
    node_patches = []
    present_nodes = set(G.nodes())
    for node in sorted(present_nodes):
        color = NODE_COLORS.get(node, NODE_COLORS["default"])
        label = node.replace('_', ' ').title()
        node_patches.append(mpatches.Patch(color=color, label=label))
    
    # Edge direction legend
    edge_patches = [
        mpatches.Patch(color=EDGE_DIRECTION_COLORS["positive"], label="Positive"),
        mpatches.Patch(color=EDGE_DIRECTION_COLORS["negative"], label="Negative"),
        mpatches.Patch(color=EDGE_DIRECTION_COLORS["ambiguous"], label="Ambiguous"),
        mpatches.Patch(color=EDGE_DIRECTION_COLORS[""], label="Unknown"),
    ]
    
    # Add node legend
    node_legend = ax.legend(
        handles=node_patches,
        loc='upper left',
        fontsize=7,
        framealpha=0.95,
        title="Economic Concepts",
        title_fontsize=8,
    )
    ax.add_artist(node_legend)
    
    # Add edge legend
    ax.legend(
        handles=edge_patches,
        loc='upper right',
        fontsize=7,
        framealpha=0.95,
        title="Causal Direction",
        title_fontsize=8,
    )


def visualize_graph(
    G: nx.DiGraph,
    period_name: str,
    output_path: Path,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
    layout_k: float = 0.5,
    show_edge_labels: bool = False,
):
    """
    Create and save a visualization of the knowledge graph.
    
    Args:
        G: NetworkX directed graph
        period_name: Human-readable period name for title
        output_path: Path to save the figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        layout_k: Spring layout spacing parameter
        show_edge_labels: Whether to show edge weights as labels
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    # Compute layout
    pos = nx.spring_layout(
        G,
        k=layout_k,
        iterations=50,
        seed=42,  # For reproducibility
    )
    
    # Get visual attributes
    node_colors = get_node_colors(G)
    node_sizes = get_node_sizes(G)
    edge_widths = get_edge_widths(G)
    edge_colors = get_edge_colors(G)
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.6,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',  # Curved edges for clarity
        ax=ax,
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=1.5,
        edgecolors='white',
        ax=ax,
    )
    
    # Draw labels for all nodes (since we have fewer now)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: node.replace('_', '\n') for node in G.nodes()},
        font_size=9,
        font_weight='bold',
        font_color='black',
        ax=ax,
    )
    
    # Optional: draw edge labels showing weights
    if show_edge_labels:
        edge_labels = {
            (u, v): f"{data['frequency']}"
            for u, v, data in G.edges(data=True)
            if data['frequency'] >= 3  # Only show frequent edges
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=6,
            ax=ax,
        )
    
    # Add title with statistics
    density = nx.density(G)
    title = f"FOMC Causal Knowledge Graph: {period_name}\n"
    title += f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Density: {density:.4f}"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legends
    create_legend(ax, G)
    
    # Clean up axes
    ax.set_axis_off()
    ax.margins(0.1)
    
    # Save figure
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Generate visualizations for all three period graphs."""
    # Setup paths
    graph_dir = ROOT / 'outputs' / 'knowledge_graphs'
    output_dir = ROOT / 'outputs'
    
    # Period configurations
    periods = {
        'great_moderation': 'Great Moderation (1994-2007)',
        'post_crisis_zlb': 'Post-Crisis ZLB (2008-2019)',
        'post_covid': 'Post-COVID Surge (2020-2023)',
    }
    
    print("=" * 70)
    print("VISUALIZING FOMC KNOWLEDGE GRAPHS")
    print("=" * 70)
    
    for period_key, period_name in periods.items():
        print(f"\nProcessing: {period_name}")
        
        # Load graph
        graph_path = graph_dir / f'graph_{period_key}.pkl'
        G = load_graph(graph_path)
        print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create visualization
        output_path = output_dir / f'kg_{period_key}.png'
        visualize_graph(G, period_name, output_path)
    
    print(f"\n{'=' * 70}")
    print("✓ Successfully generated all visualizations")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
