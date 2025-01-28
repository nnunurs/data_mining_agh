import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set
from pathlib import Path
import json
import calendar
from datetime import datetime
from clustering import ClusteringAnalyzer

class ListeningStatsAnalyzer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.artist_features = defaultdict(self._create_default_artist_features)
        self.history_data = None
        self.clustering = ClusteringAnalyzer()
    
    @staticmethod
    def _create_default_artist_features():
        return {
            'play_count': 0,
            'total_ms': 0,
            'unique_tracks': set(),
            'listening_hours': defaultdict(float),
            'genres': set(),
            'global_popularity': 0,
            'first_listen': None,
            'last_listen': None,
            'listening_gaps': [],
            'related_artists': set(),
            'top_tracks': []
        }
    
    def load_history(self) -> pd.DataFrame:
        """Loads and preprocesses streaming history"""
        all_data = []
        
        for file in self.data_dir.glob('Streaming_History_Audio_*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"No streaming history found in {self.data_dir}")
        
        self.history_data = pd.DataFrame(all_data)
        self.history_data = self.history_data[
            self.history_data['master_metadata_track_name'].notna() &
            self.history_data['master_metadata_album_artist_name'].notna()
        ]
        self.history_data['ts'] = pd.to_datetime(self.history_data['ts'])
        
        print(f"\nLoaded streaming history:")
        print(f"Total tracks: {len(self.history_data):,}")
        print(f"Unique artists: {self.history_data['master_metadata_album_artist_name'].nunique():,}")
        print(f"Date range: {self.history_data['ts'].min()} to {self.history_data['ts'].max()}")
        
        # Inicjalizuj dane artystów z historii
        unique_artists = self.history_data['master_metadata_album_artist_name'].unique()
        for artist in unique_artists:
            if artist not in self.artist_features:
                self.artist_features[artist] = self._create_default_artist_features()
        
        # Aktualizuj dane z historii
        self._update_artist_features()
        
        # Uzupełnij brakujące informacje z cache'u
        self._load_artist_cache()
        
        return self.history_data

    def _load_artist_cache(self) -> None:
        """Loads artist data from cache file"""
        cache_file = self.data_dir / 'cache' / 'artist_data.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    loaded_count = 0
                    
                    for artist in self.artist_features.keys():
                        if artist in cached_data:
                            data = cached_data[artist]
                            # Aktualizuj tylko pola, których nie ma w historii
                            try:
                                self.artist_features[artist]['global_popularity'] = float(data.get('popularity', 0))
                            except (ValueError, TypeError):
                                self.artist_features[artist]['global_popularity'] = 0
                            
                            self.artist_features[artist]['genres'] = set(data.get('genres', []))
                            self.artist_features[artist]['related_artists'] = set(data.get('related_artists', []))
                            self.artist_features[artist]['top_tracks'] = data.get('top_tracks', [])
                            loaded_count += 1
                    
                    print(f"Updated {loaded_count} artists from cache")
            except Exception as e:
                print(f"Error loading artist cache: {e}")
                if 'cached_data' in locals():
                    print(f"First problematic entry: {next(iter(cached_data.items()))}")

    def _update_artist_features(self) -> None:
        """Updates artist features based on listening history"""
        # Reset play counts and other cumulative stats
        for artist_data in self.artist_features.values():
            artist_data['play_count'] = 0
            artist_data['total_ms'] = 0
            artist_data['unique_tracks'] = set()
            artist_data['listening_hours'] = defaultdict(float)
            artist_data['listening_gaps'] = []
            artist_data['first_listen'] = None
            artist_data['last_listen'] = None
        
        # Update from history
        sorted_history = self.history_data.sort_values('ts')
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            track = row['master_metadata_track_name']
            ms_played = row['ms_played']
            ts = row['ts']
            
            artist_data = self.artist_features[artist]
            artist_data['play_count'] += 1
            artist_data['total_ms'] += ms_played
            artist_data['unique_tracks'].add(track)
            
            # Update listening hours
            hour = ts.hour
            artist_data['listening_hours'][hour] += ms_played / (1000 * 60 * 60)
            
            # Update first and last listen
            if artist_data['first_listen'] is None or ts < artist_data['first_listen']:
                artist_data['first_listen'] = ts
            if artist_data['last_listen'] is None or ts > artist_data['last_listen']:
                artist_data['last_listen'] = ts
            
            # Calculate listening gaps
            if artist_data['last_listen'] is not None and artist_data['last_listen'] != ts:
                gap = (ts - artist_data['last_listen']).total_seconds() / (60 * 60 * 24)  # days
                artist_data['listening_gaps'].append(gap)
        
        # Remove artists with no plays in history
        to_remove = [
            artist for artist, data in self.artist_features.items()
            if data['play_count'] == 0
        ]
        for artist in to_remove:
            del self.artist_features[artist]

    def analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyzes temporal listening patterns"""
        patterns = {
            'daily': defaultdict(int),
            'weekday': defaultdict(int),
            'seasonal': defaultdict(int),
            'session_length': [],
            'listening_evolution': []
        }
        
        # Daily patterns
        for hour, group in data.groupby(data['ts'].dt.hour):
            patterns['daily'][hour] = group['ms_played'].sum()
        
        # Weekly patterns
        for day, group in data.groupby(data['ts'].dt.day_name()):
            patterns['weekday'][day] = group['ms_played'].sum()
        
        # Calculate listening sessions
        # A session ends if there's more than 30 minutes between tracks
        SESSION_GAP_MINUTES = 30
        
        sorted_data = data.sort_values('ts')
        current_session = []
        current_session_length = 0
        
        for i in range(len(sorted_data)):
            if i > 0:
                time_diff = (sorted_data.iloc[i]['ts'] - sorted_data.iloc[i-1]['ts']).total_seconds() / 60
                if time_diff > SESSION_GAP_MINUTES:
                    if current_session_length > 0:
                        patterns['session_length'].append(current_session_length)
                    current_session = []
                    current_session_length = 0
            
            current_session.append(sorted_data.iloc[i])
            current_session_length += sorted_data.iloc[i]['ms_played'] / (1000 * 60)  # Convert to minutes
        
        # Don't forget the last session
        if current_session_length > 0:
            patterns['session_length'].append(current_session_length)
        
        # Monthly evolution
        monthly_data = data.groupby(pd.Grouper(key='ts', freq='M')).agg({
            'ms_played': 'sum',
            'master_metadata_album_artist_name': 'nunique'
        }).reset_index()
        
        patterns['listening_evolution'] = [
            {
                'month': row['ts'].strftime('%Y-%m'),
                'total_hours': row['ms_played'] / (1000 * 60 * 60),
                'unique_artists': row['master_metadata_album_artist_name']
            }
            for _, row in monthly_data.iterrows()
        ]
        
        return patterns

    def analyze_genre_patterns(self) -> Dict:
        """Analyzes genre-related patterns"""
        patterns = {
            'genre_expertise': defaultdict(float),
            'genre_combinations': defaultdict(int),
            'genre_flow': defaultdict(lambda: defaultdict(int))
        }
        
        # Calculate genre expertise
        for artist, features in self.artist_features.items():
            weight = features['play_count'] / max(1, len(features['genres']))
            for genre in features['genres']:
                patterns['genre_expertise'][genre] += weight
        
        # Analyze genre combinations and flow
        sorted_history = self.history_data.sort_values('ts')
        prev_genres = set()
        
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            if artist in self.artist_features:
                current_genres = self.artist_features[artist]['genres']
                
                # Genre combinations
                genre_list = sorted(list(current_genres))
                for i in range(len(genre_list)):
                    for j in range(i + 1, len(genre_list)):
                        patterns['genre_combinations'][(genre_list[i], genre_list[j])] += 1
                
                # Genre flow
                if prev_genres:
                    for from_genre in prev_genres:
                        for to_genre in current_genres:
                            if from_genre != to_genre:
                                patterns['genre_flow'][from_genre][to_genre] += 1
                
                prev_genres = current_genres
        
        return patterns

    def analyze_listening_patterns(self) -> Dict:
        """Analyzes detailed listening patterns"""
        return {
            'artist_clusters': {
                'behavior_based': self.clustering.perform_clustering(self.artist_features, "behavior"),
                'genre_based': self.clustering.perform_clustering(self.artist_features, "genre")
            },
            'listening_habits': {
                'favorite_artists': self._get_favorite_artists(),
                'discovery_rate': self._calculate_discovery_rate()
            }
        }

    def analyze_discovery_paths(self) -> Dict:
        """Analyzes music discovery patterns"""
        return {
            'discovery_sources': self._analyze_discovery_sources(),
            'artist_paths': self._find_artist_paths(),
            'exploration_patterns': self._analyze_exploration_patterns()
        }

    def get_recommendations(self, top_n: int = 10, min_plays_to_exclude: int = 10) -> Dict:
        """Generates personalized recommendations"""
        recommendations = {
            'mainstream': [],
            'hidden_gems': []
        }
        
        # Znajdź artystów do wykluczenia (często słuchanych)
        excluded_artists = {
            artist for artist, features in self.artist_features.items() 
            if features['play_count'] >= min_plays_to_exclude
        }
        
        # Znajdź aktywne gatunki z wagami
        genre_weights = defaultdict(float)
        total_plays = 0
        for artist, features in self.artist_features.items():
            if features['play_count'] > 0:
                weight = features['play_count']  # Używamy tylko liczby odtworzeń
                total_plays += features['play_count']
                for genre in features['genres']:
                    genre_weights[genre] += weight
        
        # Normalizuj wagi gatunków
        if total_plays > 0:
            for genre in genre_weights:
                genre_weights[genre] /= total_plays
        
        # Znajdź potencjalne rekomendacje
        potential_recommendations = []
        
        for artist, features in self.artist_features.items():
            if artist not in excluded_artists and features['genres']:
                # Oblicz dopasowanie gatunków
                genre_score = sum(genre_weights[genre] for genre in features['genres'])
                popularity = features['global_popularity'] / 100 if features['global_popularity'] > 0 else 0.5
                
                if genre_score > 0.01:  # Obniżamy próg dla większej liczby rekomendacji
                    potential_recommendations.append({
                        'artist': artist,
                        'genres': list(features['genres']),
                        'popularity': features['global_popularity'],
                        'genre_score': genre_score,
                        'total_score': 0.7 * genre_score + 0.3 * popularity
                    })
        
        # Sortuj i podziel rekomendacje
        mainstream_recs = [r for r in potential_recommendations if r['popularity'] >= 50]
        hidden_gems = [r for r in potential_recommendations if 20 <= r['popularity'] < 50]
        
        # Sortuj po całkowitym wyniku
        recommendations['mainstream'] = sorted(
            mainstream_recs, 
            key=lambda x: x['total_score'], 
            reverse=True
        )[:top_n]
        
        recommendations['hidden_gems'] = sorted(
            hidden_gems,
            key=lambda x: x['total_score'],
            reverse=True
        )[:top_n]
        
        return recommendations

    def print_analysis_results(self, analysis: Dict) -> None:
        """Prints analysis results in a formatted way"""
        print("\n=== Analysis Results ===")
        
        # Temporal Patterns
        print("\nTemporal Patterns:")
        if 'temporal' in analysis:
            temp = analysis['temporal']
            
            print("\nTop listening periods:")
            for day, count in sorted(temp['weekday'].items(), key=lambda x: x[1], reverse=True):
                print(f"- {day}: {count/sum(temp['weekday'].values()):.1%}")
            
            print("\nDaily distribution:")
            for hour, count in sorted(temp['daily'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"- {hour:02d}:00: {count/sum(temp['daily'].values()):.1%}")
            
            print("\nSeasonal trends:")
            quarters = defaultdict(int)
            for month, group in self.history_data.groupby(self.history_data['ts'].dt.quarter):
                quarters[f"Q{month}"] = group['ms_played'].sum()
            for quarter, count in sorted(quarters.items(), key=lambda x: x[1], reverse=True):
                print(f"- {quarter}: {count/sum(quarters.values()):.1%}")
            
            if 'session_length' in temp and temp['session_length']:  # Check if list is not empty
                print("\nListening sessions:")
                print(f"- Average length: {np.mean(temp['session_length']):.1f} minutes")
                print(f"- Longest session: {np.max(temp['session_length']):.1f} minutes")
                print(f"- Total sessions: {len(temp['session_length'])}")
        
        # Genre Analysis
        print("\nGenre Analysis:")
        if 'genres' in analysis:
            genres = analysis['genres']
            
            print("\nTop genres by expertise:")
            for genre, score in sorted(genres['genre_expertise'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
                print(f"- {genre}: {score/sum(genres['genre_expertise'].values()):.1%}")
            
            print("\nPopular genre combinations:")
            for (genre1, genre2), count in sorted(genres['genre_combinations'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]:
                print(f"- {genre1} + {genre2}: {count} times")
            
            print("\nGenre flow (most common transitions):")
            genre_flows = []
            for from_genre, transitions in genres['genre_flow'].items():
                for to_genre, count in transitions.items():
                    genre_flows.append((from_genre, to_genre, count))
            
            for from_genre, to_genre, count in sorted(genre_flows, key=lambda x: x[2], reverse=True)[:5]:
                print(f"- {from_genre} → {to_genre}: {count} times")
        
        # Discovery Patterns
        print("\nDiscovery Patterns:")
        if 'discovery' in analysis:
            discovery = analysis['discovery']
            
            print("\nDiscovery sources:")
            total_discoveries = sum(discovery['discovery_sources'].values())
            if total_discoveries > 0:  # Add this check
                for source, count in sorted(discovery['discovery_sources'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"- {source}: {count} ({count/total_discoveries:.1%})")
            else:
                print("No discoveries recorded")
            
            print("\nMost interesting discovery paths:")
            for path in sorted(discovery['artist_paths'], 
                             key=lambda x: len(x['path']), reverse=True)[:3]:
                print(f"\nPath of {len(path['path'])} artists in {path['duration_hours']:.1f} hours:")
                print(" → ".join(path['path']))
            
            if discovery['exploration_patterns']:
                recent_patterns = discovery['exploration_patterns'][-3:]
                print("\nRecent genre exploration:")
                for pattern in recent_patterns:
                    print(f"\n{pattern['month']}:")
                    print(f"- New genres discovered: {len(pattern['new_genres'])}")
                    if pattern['new_genres']:
                        print(f"- Examples: {', '.join(pattern['new_genres'][:3])}")
                    print(f"- Exploration rate: {pattern['exploration_rate']:.1%}")
        
        # Listening Patterns and Clusters
        print("\nListening Patterns:")
        if 'listening' in analysis:
            listening = analysis['listening']
            
            if 'listening_habits' in listening and listening['listening_habits']['favorite_artists']:
                recent_favorites = listening['listening_habits']['favorite_artists'][-1]
                print(f"\nTop artists ({recent_favorites['month']}):")
                for artist in recent_favorites['artists']:
                    print(f"- {artist['name']}: {artist['plays']} plays")
            
            if 'artist_clusters' in listening:
                for clustering_type, clusters in listening['artist_clusters'].items():
                    print(f"\n=== {clustering_type.title()} Based Clustering ===")
                    
                    if clusters and 'metrics' in next(iter(clusters.values())):
                        metrics = next(iter(clusters.values()))['metrics']
                        print("\nClustering metrics:")
                        print(f"- Silhouette Score: {metrics['silhouette']:.3f} (higher = better, max 1.0)")
                        print(f"- Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f} (higher = better)")
                        print(f"- Davies-Bouldin Score: {metrics['davies_bouldin']:.3f} (lower = better)")
                    
                    print(f"\nNumber of clusters: {len(clusters)}")
                    for cluster_id, cluster in clusters.items():
                        print(f"\nCluster {cluster_id} ({cluster['size']} artists):")
                        print(f"- Average popularity: {cluster['avg_popularity']:.1f}")
                        print(f"- Average plays: {cluster['avg_plays']:.1f}")
                        if 'top_genres' in cluster:
                            print(f"- Top genres: {', '.join(genre for genre, _ in cluster['top_genres'][:3])}")
                        print(f"- Example artists: {', '.join(cluster['artists'][:3])}")
        
        # Recommendations
        print("\nRecommended Artists:")
        if 'recommendations' in analysis:
            recs = analysis['recommendations']
            
            print("\nPopular Recommendations:")
            for i, rec in enumerate(recs['mainstream'][:10], 1):
                print(f"\n{i}. {rec['artist']}")
                if 'genres' in rec and rec['genres']:
                    print(f"   Genres: {', '.join(rec['genres'][:3])}")
                print(f"   Popularity: {rec['popularity']}/100")
                print(f"   Genre match: {rec['genre_score']:.2f}")
                print(f"   Overall score: {rec['total_score']:.2f}")
            
            print("\nHidden Gems (Less Popular but Matching Your Taste):")
            for i, rec in enumerate(recs['hidden_gems'][:10], 1):
                print(f"\n{i}. {rec['artist']}")
                if 'genres' in rec and rec['genres']:
                    print(f"   Genres: {', '.join(rec['genres'][:3])}")
                print(f"   Popularity: {rec['popularity']}/100")
                print(f"   Genre match: {rec['genre_score']:.2f}")
                print(f"   Overall score: {rec['total_score']:.2f}")

    def _get_favorite_artists(self) -> List[Dict]:
        """Gets favorite artists over time"""
        monthly_top = self.history_data.groupby([
            pd.Grouper(key='ts', freq='M'),
            'master_metadata_album_artist_name'
        ]).size().reset_index(name='plays')
        
        favorites = []
        for month, month_data in monthly_top.groupby(pd.Grouper(key='ts', freq='M')):
            if not month_data.empty:
                top_artists = month_data.nlargest(5, 'plays')
                favorites.append({
                    'month': month.strftime('%Y-%m'),
                    'artists': [{
                        'name': row['master_metadata_album_artist_name'],
                        'plays': row['plays']
                    } for _, row in top_artists.iterrows()]
                })
        return favorites

    def _calculate_discovery_rate(self) -> List[Dict]:
        """Calculates the rate of discovering new artists"""
        known_artists = set()
        discovery_rates = []
        
        monthly_artists = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        for month, group in monthly_artists:
            if not group.empty:
                month_artists = set(group['master_metadata_album_artist_name'])
                new_artists = month_artists - known_artists
                known_artists.update(new_artists)
                
                discovery_rates.append({
                    'month': month.strftime('%Y-%m'),
                    'new_artists': len(new_artists),
                    'total_artists': len(month_artists),
                    'discovery_rate': len(new_artists) / len(month_artists) if month_artists else 0
                })
        return discovery_rates

    def _analyze_discovery_sources(self) -> Dict[str, int]:
        """Analyzes sources of music discovery"""
        # Basic implementation counting new artists per month as discoveries
        sources = defaultdict(int)
        known_artists = set()
        
        sorted_history = self.history_data.sort_values('ts')
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            if artist not in known_artists:
                # Simplified logic - treat all new artists as "new_discovery"
                sources['new_discovery'] += 1
                known_artists.add(artist)
        
        # Add at least one item to avoid division by zero
        if not sources:
            sources['new_discovery'] = 0
        
        return dict(sources)

    def _find_artist_paths(self) -> List[Dict]:
        """Finds paths of artist discovery"""
        paths = []
        min_path_length = 5  # minimum artists in a path
        max_path_length = 25  # zmniejszamy z 50 do 25
        max_time_gap_hours = 2
        min_play_time = 30  # minimum seconds played to count
        
        sorted_history = self.history_data[
            self.history_data['ms_played'] >= (min_play_time * 1000)  # filtrujemy krótkie odtworzenia
        ].sort_values('ts')
        
        current_path = []
        current_duration = 0
        
        for i in range(len(sorted_history)):
            if i > 0:
                time_diff = (sorted_history.iloc[i]['ts'] - sorted_history.iloc[i-1]['ts']).total_seconds() / 3600
                if time_diff > max_time_gap_hours:
                    if len(current_path) >= min_path_length:
                        if len(current_path) > max_path_length:
                            current_path = current_path[:max_path_length]
                        # Dodajemy sprawdzenie unikalności
                        unique_ratio = len(set(current_path)) / len(current_path)
                        if unique_ratio >= 0.8:  # minimum 80% unikalnych artystów
                            paths.append({
                                'path': current_path,
                                'duration_hours': current_duration,
                                'unique_ratio': unique_ratio
                            })
                    current_path = []
                    current_duration = 0
            
            artist = sorted_history.iloc[i]['master_metadata_album_artist_name']
            if not current_path or (
                artist != current_path[-1] and 
                current_path.count(artist) < 2  # max 1 powtórzenie
            ):
                current_path.append(artist)
            current_duration += sorted_history.iloc[i]['ms_played'] / (1000 * 60 * 60)
        
        # Sortujemy po unikalności i długości
        paths.sort(key=lambda x: (
            x['unique_ratio'],
            len(x['path']),
            -x['duration_hours']  # krótsze ścieżki preferowane przy tym samym unique_ratio
        ), reverse=True)
        
        return paths[:3]  # tylko top 3 ścieżki

    def _analyze_exploration_patterns(self) -> List[Dict]:
        """Analyzes patterns in music exploration"""
        patterns = []
        monthly_data = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        
        known_genres = set()
        for month, group in monthly_data:
            if not group.empty:
                month_artists = group['master_metadata_album_artist_name'].unique()
                month_genres = set()
                for artist in month_artists:
                    if artist in self.artist_features:
                        month_genres.update(self.artist_features[artist]['genres'])
                
                new_genres = month_genres - known_genres
                known_genres.update(new_genres)
                
                patterns.append({
                    'month': month.strftime('%Y-%m'),
                    'new_genres': list(new_genres),
                    'total_genres': len(month_genres),
                    'exploration_rate': len(new_genres) / len(month_genres) if month_genres else 0
                })
        
        return patterns 