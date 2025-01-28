from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import warnings
from typing import Dict

from spotify_client import SpotifyClient
from listening_stats import ListeningStatsAnalyzer
from clustering import ClusteringAnalyzer
from visualization import Visualizer, DataVisualizer

warnings.filterwarnings('ignore')
load_dotenv()

class SpotifyAnalyzer:
    """Main Spotify music analyzer and recommender."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / 'cache'
        self.plots_dir = self.data_dir / 'plots'
        
        self.cache_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        self.spotify_client = SpotifyClient()
        self.stats_analyzer = ListeningStatsAnalyzer(self.data_dir)
        self.clustering = ClusteringAnalyzer()
        self.visualizer = Visualizer(self.plots_dir)
        
        self.history_data = None
        self.load_streaming_history()

    def load_streaming_history(self) -> None:
        """Loads streaming history from Spotify files"""
        self.history_data = self.stats_analyzer.load_history()

    def analyze_all(self) -> Dict:
        """Performs comprehensive music analysis"""
        print("\nRunning comprehensive music analysis...")
        
        analysis = {
            'temporal': self.stats_analyzer.analyze_temporal_patterns(self.history_data),
            'genres': self.stats_analyzer.analyze_genre_patterns(),
            'listening': self.stats_analyzer.analyze_listening_patterns(),
            'discovery': self.stats_analyzer.analyze_discovery_paths(),
            'recommendations': self.stats_analyzer.get_recommendations()
        }
        
        behavior_clusters = self.clustering.perform_clustering(self.stats_analyzer.artist_features, "behavior")
        features, artists = self.clustering.prepare_features(self.stats_analyzer.artist_features, "behavior")
        
        visualizer = DataVisualizer(self.plots_dir)
        visualizer.visualize_clusters(behavior_clusters, features, artists, "Behavior-Based Clustering")
        
        return analysis

if __name__ == "__main__":
    analyzer = SpotifyAnalyzer()
    analysis = analyzer.analyze_all()
    
    # Generate visualizations
    analyzer.visualizer.generate_all_visualizations(analysis)
    
    # Print analysis results
    analyzer.stats_analyzer.print_analysis_results(analysis) 