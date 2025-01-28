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

    def analyze_artists(self) -> Dict:
        """Analize artist statistics"""
        print("\nAnalyzing artist statistics...")
        
        analysis = {
            'top_artists': [],
            'artist_clusters': self._cluster_artists(),
            'listening_moods': self._analyze_listening_moods(),
            'artist_relationships': self._analyze_artist_relationships(),
            'listening_habits': {
                'favorite_artists': [],
                'discovery_rate': [],
                'repeat_patterns': defaultdict(int),
                'listening_streaks': []
            }
        }
        
        # Top artists analysis
        top_artists = sorted(
            self.artist_features.items(),
            key=lambda x: x[1]['play_count'],
            reverse=True
        )[:50]
        
        for artist, data in top_artists:
            analysis['top_artists'].append({
                'name': artist,
                'play_count': data['play_count'],
                'total_time': data['total_ms'] / (1000 * 60 * 60),  # w godzinach
                'unique_tracks': len(data['unique_tracks']),
                'genres': list(data['genres']),
                'popularity': data['global_popularity']
            })
        
        # Favorite artists analysis over time
        monthly_top = self.history_data.groupby([
            pd.Grouper(key='ts', freq='M'),
            'master_metadata_album_artist_name'
        ]).size().reset_index(name='plays')
        
        for month, month_data in monthly_top.groupby(pd.Grouper(key='ts', freq='M')):
            if not month_data.empty:
                top_month_artists = month_data.nlargest(5, 'plays')
                analysis['listening_habits']['favorite_artists'].append({
                    'month': month.strftime('%Y-%m'),
                    'artists': [{
                        'name': row['master_metadata_album_artist_name'],
                        'plays': row['plays']
                    } for _, row in top_month_artists.iterrows()]
                })
        
        # New artists discovery analysis
        known_artists = set()
        monthly_discovery = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        
        for month, group in monthly_discovery:
            month_artists = set(group['master_metadata_album_artist_name'])
            new_artists = month_artists - known_artists
            known_artists.update(new_artists)
            
            analysis['listening_habits']['discovery_rate'].append({
                'month': month.strftime('%Y-%m'),
                'new_artists': len(new_artists),
                'total_artists': len(month_artists),
                'discovery_rate': len(new_artists) / len(month_artists) if month_artists else 0
            })
        
        # Repeat patterns analysis
        for artist, features in self.artist_features.items():
            analysis['listening_habits']['repeat_patterns'][features['play_count']] += 1
        
        # Listening streaks analysis
        current_streak = []
        prev_date = None
        
        for date in sorted(self.history_data['ts'].dt.date.unique()):
            if prev_date is None or (date - prev_date).days == 1:
                current_streak.append(date)
            else:
                if len(current_streak) >= 3:
                    analysis['listening_habits']['listening_streaks'].append({
                        'start': current_streak[0].isoformat(),
                        'end': current_streak[-1].isoformat(),
                        'length': len(current_streak)
                    })
                current_streak = [date]
            prev_date = date
        
        return analysis

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
