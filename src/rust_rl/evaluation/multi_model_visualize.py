"""
Visualization utilities for multi-model comparison
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not available, using matplotlib default colors")
    sns = None

from .config import UnifiedConfig


class MultiModelVisualizer:
    """Creates visualizations comparing multiple models"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.output_dir = Path(config.output_base_dir) / "comparison_charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        if sns is not None:
            sns.set_palette("husl")
    
    def create_all_visualizations(self, results_dict: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """
        Create all visualization charts
        
        Args:
            results_dict: Dictionary mapping model names to their results DataFrames
            
        Returns:
            Dictionary mapping chart names to their file paths
        """
        if not results_dict:
            print("No results provided for visualization")
            return {}
        
        chart_paths = {}
        
        # Overall success rates
        chart_paths["overall_success"] = self.plot_overall_success_rates(results_dict)
        
        # Tool-specific performance
        chart_paths["tool_performance"] = self.plot_tool_specific_performance(results_dict)
        
        # Success rate heatmap
        chart_paths["success_heatmap"] = self.plot_success_rate_heatmap(results_dict)
        
        # Performance breakdown
        chart_paths["performance_breakdown"] = self.plot_performance_breakdown(results_dict)
        
        # Create summary statistics
        chart_paths["summary_csv"] = self.create_summary_statistics(results_dict)
        
        return chart_paths
    
    def plot_overall_success_rates(self, results_dict: Dict[str, pd.DataFrame]) -> Path:
        """Plot overall success rates for all models"""
        model_names = []
        success_rates = []
        
        for model_name, results_df in results_dict.items():
            # Calculate overall success rate (all tools pass)
            all_tools_pass = True
            for tool in self.config.evaluation_tools:
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    all_tools_pass = all_tools_pass & results_df[tool_col]
            
            success_rate = all_tools_pass.mean() * 100
            model_names.append(model_name)
            success_rates.append(success_rate)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("husl", len(model_names)) if sns else plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, success_rates, color=colors)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Overall Success Rate (All Tools Pass)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "overall_success_rates.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_tool_specific_performance(self, results_dict: Dict[str, pd.DataFrame]) -> Path:
        """Plot performance for each tool across all models"""
        # Prepare data
        plot_data = []
        
        for model_name, results_df in results_dict.items():
            for tool in self.config.evaluation_tools:
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    success_rate = results_df[tool_col].mean() * 100
                    plot_data.append({
                        'Model': model_name,
                        'Tool': tool.capitalize(),
                        'Success Rate': success_rate
                    })
        
        if not plot_data:
            print("No tool performance data available")
            return None
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot data for grouped bar plot
        pivot_df = plot_df.pivot(index='Model', columns='Tool', values='Success Rate')
        
        # Create grouped bar plot
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Tool-Specific Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(title='Tool', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "tool_specific_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_success_rate_heatmap(self, results_dict: Dict[str, pd.DataFrame]) -> Path:
        """Create a heatmap of success rates across models and tools"""
        # Prepare data matrix
        models = list(results_dict.keys())
        tools = [tool.capitalize() for tool in self.config.evaluation_tools]
        
        success_matrix = np.zeros((len(models), len(tools)))
        
        for i, model_name in enumerate(models):
            results_df = results_dict[model_name]
            for j, tool in enumerate(self.config.evaluation_tools):
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    success_matrix[i, j] = results_df[tool_col].mean() * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(tools)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(tools)
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(tools)):
                text = ax.text(j, i, f'{success_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Success Rate Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tool', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "success_rate_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_performance_breakdown(self, results_dict: Dict[str, pd.DataFrame]) -> Path:
        """Plot detailed performance breakdown for each model"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results_df) in enumerate(results_dict.items()):
            ax = axes[i]
            
            # Calculate success rates for each tool
            tool_rates = []
            tool_names = []
            
            for tool in self.config.evaluation_tools:
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    rate = results_df[tool_col].mean() * 100
                    tool_rates.append(rate)
                    tool_names.append(tool.capitalize())
            
            # Create pie chart
            if tool_rates:
                colors = sns.color_palette("husl", len(tool_rates)) if sns else plt.cm.tab10(np.linspace(0, 1, len(tool_rates)))
                wedges, texts, autotexts = ax.pie(tool_rates, labels=tool_names, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
                
                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            ax.set_title(f'{model_name}\nPerformance Breakdown', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "performance_breakdown.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_summary_statistics(self, results_dict: Dict[str, pd.DataFrame]) -> Path:
        """Create summary statistics CSV"""
        summary_data = []
        
        for model_name, results_df in results_dict.items():
            row = {'Model': model_name}
            
            # Individual tool success rates
            for tool in self.config.evaluation_tools:
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    rate = results_df[tool_col].mean() * 100
                    row[f'{tool.capitalize()}_Success_Rate'] = f'{rate:.2f}%'
            
            # Overall success rate (all tools pass)
            all_tools_pass = True
            for tool in self.config.evaluation_tools:
                tool_col = f"{tool}_passed"
                if tool_col in results_df.columns:
                    all_tools_pass = all_tools_pass & results_df[tool_col]
            
            overall_rate = all_tools_pass.mean() * 100
            row['Overall_Success_Rate'] = f'{overall_rate:.2f}%'
            row['Total_Samples'] = len(results_df)
            
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        output_path = self.output_dir / "model_comparison_summary.csv"
        summary_df.to_csv(output_path, index=False)
        
        return output_path