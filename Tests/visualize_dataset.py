#!/usr/bin/env python3
"""
METR-LA Traffic Dataset Visualization
=====================================
This script provides comprehensive visualizations for the METR-LA traffic dataset
including network topology, traffic patterns, and temporal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime, timedelta

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the METR-LA dataset."""
    print("Loading METR-LA dataset...")
    adj_mat = np.load('./dataset/adj_mat.npy')
    node_values = np.load('./dataset/node_values.npy')
    
    print(f"Adjacency Matrix: {adj_mat.shape}")
    print(f"Node Values: {node_values.shape}")
    print(f"- {adj_mat.shape[0]} traffic sensors")
    print(f"- {node_values.shape[0]} time steps")
    print(f"- {node_values.shape[2]} features per sensor")
    
    return adj_mat, node_values

def visualize_network_topology(adj_mat, save_path=None):
    """Visualize the traffic sensor network topology."""
    print("Creating network topology visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('METR-LA Traffic Network Analysis', fontsize=16, fontweight='bold')
    
    # 1. Adjacency Matrix Heatmap
    ax1 = axes[0, 0]
    sns.heatmap(adj_mat, cmap='Blues', ax=ax1, cbar_kws={'label': 'Connection Weight'})
    ax1.set_title('Adjacency Matrix Heatmap')
    ax1.set_xlabel('Sensor ID')
    ax1.set_ylabel('Sensor ID')
    
    # 2. Network Graph
    ax2 = axes[0, 1]
    G = nx.from_numpy_array(adj_mat)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='lightblue', 
                          alpha=0.7, ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax2)
    ax2.set_title(f'Traffic Sensor Network\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    ax2.axis('off')
    
    # 3. Degree Distribution
    ax3 = axes[1, 0]
    degrees = [G.degree(n) for n in G.nodes()]
    ax3.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Node Degree Distribution')
    ax3.set_xlabel('Degree')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Network Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate network statistics
    avg_degree = np.mean(degrees)
    density = nx.density(G)
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
    
    stats_text = f"""Network Statistics:
    
• Nodes: {G.number_of_nodes()}
• Edges: {G.number_of_edges()}
• Density: {density:.4f}
• Avg Degree: {avg_degree:.2f}
• Max Degree: {max(degrees)}
• Avg Clustering: {avg_clustering:.4f}
• Sparsity: {(1 - np.count_nonzero(adj_mat) / adj_mat.size) * 100:.2f}%
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return G

def visualize_traffic_patterns(node_values, save_path=None):
    """Visualize traffic patterns and temporal analysis."""
    print("Creating traffic pattern visualizations...")
    
    # Extract traffic data (assuming first feature is traffic flow)
    traffic_data = node_values[:, :, 0]  # Shape: (time_steps, sensors)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Traffic Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Traffic heatmap over time (sample)
    ax1 = axes[0, 0]
    sample_data = traffic_data[:1000:10, :50]  # Sample every 10th timestep, first 50 sensors
    im1 = ax1.imshow(sample_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Traffic Intensity Heatmap\n(Sample: First 50 sensors)')
    ax1.set_xlabel('Time Steps (sampled)')
    ax1.set_ylabel('Sensor ID')
    plt.colorbar(im1, ax=ax1, label='Traffic Flow')
    
    # 2. Average traffic by sensor
    ax2 = axes[0, 1]
    avg_traffic = np.mean(traffic_data, axis=0)
    ax2.bar(range(len(avg_traffic)), avg_traffic, alpha=0.7, color='coral')
    ax2.set_title('Average Traffic by Sensor')
    ax2.set_xlabel('Sensor ID')
    ax2.set_ylabel('Average Traffic Flow')
    ax2.grid(True, alpha=0.3)
    
    # 3. Traffic distribution
    ax3 = axes[0, 2]
    ax3.hist(traffic_data.flatten(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Traffic Flow Distribution')
    ax3.set_xlabel('Traffic Flow')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series for selected sensors
    ax4 = axes[1, 0]
    selected_sensors = [0, 50, 100, 150, 200]  # Sample sensors
    sample_timesteps = slice(0, 2000)  # First 2000 timesteps
    
    for sensor in selected_sensors:
        if sensor < traffic_data.shape[1]:
            ax4.plot(traffic_data[sample_timesteps, sensor], 
                    label=f'Sensor {sensor}', alpha=0.8, linewidth=1)
    
    ax4.set_title('Traffic Time Series (Selected Sensors)')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Traffic Flow')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Daily patterns (if we can infer daily cycles)
    ax5 = axes[1, 1]
    # Assume 5-minute intervals, so 288 steps per day
    steps_per_day = 288
    if traffic_data.shape[0] >= steps_per_day:
        # Average traffic by time of day
        daily_patterns = []
        for day_start in range(0, traffic_data.shape[0] - steps_per_day, steps_per_day):
            day_data = traffic_data[day_start:day_start + steps_per_day]
            daily_patterns.append(np.mean(day_data, axis=1))
        
        if daily_patterns:
            daily_avg = np.mean(daily_patterns, axis=0)
            hours = np.arange(len(daily_avg)) * 5 / 60  # Convert to hours
            ax5.plot(hours, daily_avg, color='purple', linewidth=2)
            ax5.set_title('Average Daily Traffic Pattern')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Average Traffic Flow')
            ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data\nfor daily pattern analysis', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Daily Pattern (Not Available)')
    
    # 6. Correlation matrix (sample of sensors)
    ax6 = axes[1, 2]
    sample_sensors_for_corr = min(20, traffic_data.shape[1])
    corr_data = traffic_data[:5000, :sample_sensors_for_corr]  # Sample data for correlation
    correlation_matrix = np.corrcoef(corr_data.T)
    
    im6 = ax6.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_title(f'Sensor Correlation Matrix\n(First {sample_sensors_for_corr} sensors)')
    ax6.set_xlabel('Sensor ID')
    ax6.set_ylabel('Sensor ID')
    plt.colorbar(im6, ax=ax6, label='Correlation')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(adj_mat, node_values):
    """Create a summary of dataset statistics."""
    print("\n" + "="*50)
    print("METR-LA Dataset Summary")
    print("="*50)
    
    traffic_data = node_values[:, :, 0]
    
    print(f"📊 Dataset Dimensions:")
    print(f"   • Time steps: {node_values.shape[0]:,}")
    print(f"   • Sensors: {node_values.shape[1]}")
    print(f"   • Features per sensor: {node_values.shape[2]}")
    
    print(f"\n🚗 Traffic Statistics:")
    print(f"   • Min flow: {traffic_data.min():.2f}")
    print(f"   • Max flow: {traffic_data.max():.2f}")
    print(f"   • Mean flow: {traffic_data.mean():.2f}")
    print(f"   • Std deviation: {traffic_data.std():.2f}")
    
    print(f"\n🌐 Network Statistics:")
    print(f"   • Nodes: {adj_mat.shape[0]}")
    print(f"   • Connections: {np.count_nonzero(adj_mat):,}")
    print(f"   • Network density: {np.count_nonzero(adj_mat) / (adj_mat.shape[0]**2):.4f}")
    print(f"   • Sparsity: {(1 - np.count_nonzero(adj_mat) / adj_mat.size) * 100:.2f}%")
    
    # Time span estimation (assuming 5-minute intervals)
    total_minutes = node_values.shape[0] * 5
    days = total_minutes / (24 * 60)
    print(f"\n⏰ Temporal Coverage:")
    print(f"   • Total time steps: {node_values.shape[0]:,}")
    print(f"   • Estimated duration: {days:.1f} days")
    print(f"   • Assuming 5-minute intervals")

def main():
    """Main function to run all visualizations."""
    print("🚦 METR-LA Traffic Dataset Visualization Tool")
    print("=" * 50)
    
    # Load data
    adj_mat, node_values = load_data()
    
    # Create summary statistics
    create_summary_statistics(adj_mat, node_values)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    G = visualize_network_topology(adj_mat, save_path='network_topology.png')
    visualize_traffic_patterns(node_values, save_path='traffic_patterns.png')
    
    print("\n✅ Visualization complete!")
    print("📁 Files saved:")
    print("   • network_topology.png")
    print("   • traffic_patterns.png")

if __name__ == "__main__":
    main()
