import json
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import pylast
from dotenv import load_dotenv

class MusicRecommender:
    def __init__(self):
        self.history_data = None
        self.artist_features = defaultdict(lambda: {
            'play_count': 0,
            'total_ms': 0,
            'unique_tracks': set(),
            'listening_hours': defaultdict(float),
            'tags': set(),
            'similar_artists': set(),
            'global_listeners': 0
        })
        self.network = None
        self.favorite_tags = None
        self.preferred_hours = None
        self.artist_clusters = None
        self.scaler = StandardScaler()
        
    def setup_lastfm_client(self):
        load_dotenv()
        api_key = os.getenv('LASTFM_API_KEY')
        api_secret = os.getenv('LASTFM_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Missing Last.fm API keys. Set LASTFM_API_KEY and LASTFM_API_SECRET in .env file")
        
        self.network = pylast.LastFMNetwork(
            api_key=api_key,
            api_secret=api_secret
        )
        
        print("Successfully connected to Last.fm API!")
    
    def load_streaming_history(self, data_dir='data'):
        all_data = []
        data_path = Path(data_dir)
        
        for file in data_path.glob('Streaming_History_Audio_*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        self.history_data = pd.DataFrame(all_data)
        self.history_data = self.history_data[self.history_data['master_metadata_track_name'].notna()]
        
        print("\nLoaded data statistics:")
        print(f"Total plays: {len(self.history_data)}")
        print(f"Unique artists: {self.history_data['master_metadata_album_artist_name'].nunique()}")
        
        return self.history_data
    
    def _retry_api_call(self, func, *args, max_retries=5, base_delay=1.0, **kwargs):
        last_error = None
        
        for attempt in range(max_retries):
            try:
                time.sleep(base_delay * (1.2 ** attempt))
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
        print(f"All attempts failed. Last error: {last_error}")
        return None
    
    def get_artist_features(self, artist_name):
        try:
            artist = self.network.get_artist(artist_name)
            
            tags = self._retry_api_call(artist.get_top_tags, limit=10)
            artist_tags = {tag.item.get_name().lower() for tag in tags}
            
            similar = self._retry_api_call(artist.get_similar, limit=10)
            similar_artists = set()
            if similar:
                for similar_artist in similar:
                    try:
                        if hasattr(similar_artist, 'name'):
                            similar_artists.add(similar_artist.name)
                        elif hasattr(similar_artist, 'item'):
                            similar_artists.add(similar_artist.item.name)
                        elif isinstance(similar_artist, str):
                            similar_artists.add(similar_artist)
                    except Exception:
                        continue
            
            listeners = self._retry_api_call(artist.get_listener_count)
            
            return {
                'tags': artist_tags,
                'similar_artists': similar_artists,
                'global_listeners': listeners if listeners else 0
            }
            
        except Exception as e:
            print(f"Error fetching info for artist {artist_name}: {e}")
            return None
    
    def process_listening_history(self):
        if self.history_data is None:
            raise ValueError("No data loaded. Call load_streaming_history() first")
        
        self.history_data['ts'] = pd.to_datetime(self.history_data['ts'])
        
        print("Analyzing listening history...")
        for _, row in tqdm(self.history_data.iterrows(), total=len(self.history_data), desc="Collecting basic statistics"):
            artist = row['master_metadata_album_artist_name']
            track = row['master_metadata_track_name']
            hour = row['ts'].hour
            ms_played = row['ms_played']
            
            if ms_played < 30000:
                continue
                
            self.artist_features[artist]['play_count'] += 1
            self.artist_features[artist]['total_ms'] += ms_played
            self.artist_features[artist]['unique_tracks'].add(track)
            self.artist_features[artist]['listening_hours'][hour] += ms_played / 3600000
        
        top_artists = sorted(
            self.artist_features.items(),
            key=lambda x: x[1]['play_count'],
            reverse=True
        )[:100]
        
        print("\nFetching artist information from Last.fm...")
        for artist_name, data in tqdm(top_artists, desc="Fetching API data"):
            features = self.get_artist_features(artist_name)
            if features:
                data.update(features)
            time.sleep(0.25)
        
        self._analyze_listening_patterns()
        self._cluster_artists()
    
    def _analyze_listening_patterns(self):
        all_tags = defaultdict(int)
        for artist, data in self.artist_features.items():
            weight = np.log1p(data['play_count'])
            for tag in data.get('tags', set()):
                all_tags[tag] += weight
        
        self.favorite_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]
        
        hours_distribution = defaultdict(float)
        for data in self.artist_features.values():
            for hour, duration in data['listening_hours'].items():
                hours_distribution[hour] += duration
        
        total_hours = sum(hours_distribution.values())
        self.preferred_hours = {hour: duration/total_hours 
                              for hour, duration in hours_distribution.items()}
    
    def _cluster_artists(self):
        print("\nStarting artist clustering...")
        
        tag_counts = defaultdict(int)
        artists_with_tags = 0
        ignored_tags = {'seen live', 'spotify', 'favorite', 'favourites', 'my favorite', 'my favourites'}
        
        for data in self.artist_features.values():
            if 'tags' in data and data['tags']:
                artists_with_tags += 1
                normalized_tags = {tag.strip().lower() for tag in data['tags']}
                normalized_tags = {tag for tag in normalized_tags 
                                 if len(tag) > 1 and tag not in ignored_tags}
                data['tags'] = normalized_tags
                for tag in normalized_tags:
                    tag_counts[tag] += 1
        
        print(f"Found {len(tag_counts)} unique tags")
        print(f"Artists with tags: {artists_with_tags}")
        
        min_occurrences = max(3, int(artists_with_tags * 0.03))
        max_occurrences = int(artists_with_tags * 0.6)
        
        filtered_tags = {
            tag for tag, count in tag_counts.items() 
            if min_occurrences <= count <= max_occurrences
        }
        
        print(f"\nTag filtering criteria:")
        print(f"- Minimum occurrences: {min_occurrences}")
        print(f"- Maximum occurrences: {max_occurrences}")
        print(f"Tags after filtering: {len(filtered_tags)}")
        
        if filtered_tags:
            print("\nMost common tags after filtering:")
            top_tags = sorted(
                [(tag, count) for tag, count in tag_counts.items() if tag in filtered_tags],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for tag, count in top_tags:
                print(f"- {tag}: {count} occurrences")
        
        tag_to_idx = {tag: i for i, tag in enumerate(filtered_tags)}
        
        features = []
        artists = []
        
        for artist, data in self.artist_features.items():
            play_count_norm = np.log1p(data['play_count'])
            unique_tracks_norm = len(data['unique_tracks']) / max(1, data['play_count'])
            avg_track_length = data['total_ms'] / (max(1, data['play_count']) * 1000)
            total_time = data['total_ms'] / (1000 * 60 * 60)
            
            numeric_features = [
                play_count_norm,
                unique_tracks_norm,
                avg_track_length / 300,
                np.log1p(total_time)
            ]
            
            if filtered_tags:
                tag_vector = np.zeros(len(tag_to_idx))
                artist_tags = data.get('tags', set()) & filtered_tags
                for tag in artist_tags:
                    tag_vector[tag_to_idx[tag]] = 1
                tag_vector = tag_vector * 2
                features.append(np.append(tag_vector, numeric_features))
            else:
                features.append(np.array(numeric_features))
            
            artists.append(artist)
        
        if len(features) < 5:
            print("\nWarning: Too few artists for clustering")
            self.artist_clusters = None
            return
        
        features = np.array(features)
        print(f"\nPrepared clustering data for {len(features)} artists")
        print(f"Features per artist: {features.shape[1]}")
        
        features = self.scaler.fit_transform(features)
        
        max_clusters = min(8, len(features) // 100 + 3)
        inertias = []
        
        for k in range(3, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        if len(inertias) > 1:
            inertia_changes = np.diff(inertias)
            optimal_clusters = np.argmin(np.abs(inertia_changes)) + 4
        else:
            optimal_clusters = 3
        
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        self.artist_clusters = {
            'centers': kmeans.cluster_centers_,
            'artists': list(zip(artists, clusters)),
            'features': features,
            'tag_to_idx': tag_to_idx
        }
        
        cluster_stats = defaultdict(lambda: {
            'count': 0,
            'artists': [],
            'avg_listeners': 0,
            'avg_tracks': 0,
            'avg_playtime': 0,
            'top_tags': defaultdict(int)
        })
        
        for artist, cluster in zip(artists, clusters):
            data = self.artist_features[artist]
            stats = cluster_stats[cluster]
            stats['count'] += 1
            stats['artists'].append(artist)
            stats['avg_listeners'] += data.get('global_listeners', 0)
            stats['avg_tracks'] += len(data['unique_tracks'])
            stats['avg_playtime'] += data['total_ms'] / (1000 * 60 * 60)
            for tag in data.get('tags', set()):
                if tag not in ignored_tags:
                    stats['top_tags'][tag] += 1
        
        for stats in cluster_stats.values():
            if stats['count'] > 0:
                stats['avg_listeners'] /= stats['count']
                stats['avg_tracks'] /= stats['count']
                stats['avg_playtime'] /= stats['count']
        
        print(f"\nCreated {optimal_clusters} clusters:")
        for cluster_id, stats in sorted(cluster_stats.items(), 
                                      key=lambda x: (x[1]['avg_playtime'], x[1]['count']), 
                                      reverse=True):
            print(f"\nCluster {cluster_id}:")
            print(f"Number of artists: {stats['count']}")
            print(f"Average listeners: {stats['avg_listeners']:,.0f}")
            print(f"Average tracks: {stats['avg_tracks']:.1f}")
            print(f"Average listening time: {stats['avg_playtime']:.1f} hours")
            
            if stats['top_tags']:
                top_tags = sorted(stats['top_tags'].items(), key=lambda x: x[1], reverse=True)[:3]
                print("Most common tags:", ", ".join(f"{tag} ({count})" for tag, count in top_tags))
            
            print("Example artists:", ", ".join(stats['artists'][:3]))
    
    def get_recommendations(self, top_n=5):
        if not hasattr(self, 'artist_features') or not self.artist_features:
            raise ValueError("Process data first using process_listening_history()")
        
        print("\nYour music preferences:")
        if self.favorite_tags:
            print("\nFavorite tags:")
            for tag, weight in self.favorite_tags[:5]:
                print(f"- {tag}")
        
        if self.preferred_hours:
            print("\nPreferred listening hours:")
            peak_hours = sorted(self.preferred_hours.items(), key=lambda x: x[1], reverse=True)[:3]
            for hour, percentage in peak_hours:
                print(f"- {hour}:00 ({percentage*100:.1f}% of listening)")
        
        candidates = set()
        for artist, data in self.artist_features.items():
            if 'similar_artists' in data:
                candidates.update(data['similar_artists'])
        
        candidates = candidates - set(self.artist_features.keys())
        
        if not candidates:
            print("\nWarning: No recommendation candidates found")
            return []
        
        print(f"\nFound {len(candidates)} potential artists for recommendations")
        
        recommendations = []
        for artist_name in tqdm(list(candidates)[:50], desc="Analyzing potential recommendations"):
            features = self.get_artist_features(artist_name)
            if features and features.get('tags'):
                tag_similarity = len(features['tags'] & {tag for tag, _ in self.favorite_tags}) / len(self.favorite_tags) if self.favorite_tags else 0
                popularity_score = np.log1p(features['global_listeners']) / 20
                final_score = 0.7 * tag_similarity + 0.3 * popularity_score
                
                recommendations.append({
                    'artist': artist_name,
                    'tags': list(features['tags']),
                    'listeners': features['global_listeners'],
                    'tag_similarity': tag_similarity,
                    'score': final_score
                })
            
            time.sleep(0.25)
        
        if not recommendations:
            print("\nWarning: Could not find suitable recommendations")
            return []
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]

if __name__ == "__main__":
    recommender = MusicRecommender()
    
    print("Configuring Last.fm API access...")
    recommender.setup_lastfm_client()
    
    print("Loading listening history...")
    recommender.load_streaming_history()
    
    print("Analyzing listening patterns...")
    recommender.process_listening_history()
    
    print("\nSearching for new artists...")
    recommendations = recommender.get_recommendations()
    
    print("\nRecommended new artists:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['artist']}")
        print(f"   - Tags: {', '.join(rec['tags'][:3])}")
        print(f"   - Listeners: {rec['listeners']:,}")
        print(f"   - Tag similarity: {rec['tag_similarity']:.2f}")
        print(f"   - Overall match score: {rec['score']:.2f}") 