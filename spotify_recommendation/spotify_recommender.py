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
from typing import Dict, List

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
        """Przetwarza historię słuchania i zbiera informacje o artystach"""
        if self.history_data is None or self.network is None:
            raise ValueError("Load history data and setup Last.fm client first")

        print("\nProcessing listening history...")
        
        # Wczytaj cache gatunków jeśli istnieje
        cache_file = Path('data/artist_cache.json')
        cache_file.parent.mkdir(exist_ok=True)
        
        artist_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    artist_cache = json.load(f)
                print(f"Loaded {len(artist_cache)} artists from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Zbierz podstawowe statystyki
        for _, row in tqdm(self.history_data.iterrows(), desc="Processing plays"):
            artist_name = row['master_metadata_album_artist_name']
            track_name = row['master_metadata_track_name']
            ms_played = row['ms_played']
            hour = pd.to_datetime(row['ts']).hour
            
            self.artist_features[artist_name]['play_count'] += 1
            self.artist_features[artist_name]['total_ms'] += ms_played
            self.artist_features[artist_name]['unique_tracks'].add(track_name)
            self.artist_features[artist_name]['listening_hours'][hour] += ms_played / (1000 * 60 * 60)

        # Znajdź artystów, których nie ma w cache
        uncached_artists = [artist for artist in self.artist_features.keys() 
                           if artist not in artist_cache]
        
        if uncached_artists:
            print(f"\nFetching data for {len(uncached_artists)} new artists...")
            
            for artist_name in tqdm(uncached_artists, desc="Fetching artist data"):
                try:
                    artist = self.network.get_artist(artist_name)
                    
                    # Pobierz tagi
                    tags = self._retry_api_call(artist.get_top_tags)
                    if tags:
                        artist_tags = [tag.item.name.lower() for tag in tags[:10]]
                    else:
                        artist_tags = []
                    
                    # Pobierz liczbę słuchaczy
                    listeners = self._retry_api_call(artist.get_listener_count)
                    
                    # Zapisz do cache'u
                    artist_cache[artist_name] = {
                        'tags': artist_tags,
                        'listeners': listeners if listeners else 0
                    }
                    
                    # Aktualizuj features
                    self.artist_features[artist_name]['tags'].update(artist_tags)
                    self.artist_features[artist_name]['global_listeners'] = listeners if listeners else 0
                    
                except Exception as e:
                    print(f"Error processing {artist_name}: {e}")
                    continue
            
            # Zapisz zaktualizowany cache
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(artist_cache, f, ensure_ascii=False, indent=2)
                print(f"Updated cache with {len(artist_cache)} artists")
            except Exception as e:
                print(f"Error saving cache: {e}")
        else:
            # Użyj danych z cache'u
            print("\nUsing cached artist data...")
            for artist_name, cache_data in artist_cache.items():
                if artist_name in self.artist_features:
                    self.artist_features[artist_name]['tags'].update(cache_data['tags'])
                    self.artist_features[artist_name]['global_listeners'] = cache_data['listeners']

        # Zbierz wszystkie unikalne tagi
        all_tags = set()
        for features in self.artist_features.values():
            all_tags.update(features['tags'])
        
        # Znajdź najczęstsze tagi
        tag_counts = defaultdict(int)
        for features in self.artist_features.values():
            for tag in features['tags']:
                tag_counts[tag] += features['play_count']
        
        self.favorite_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        # Oblicz preferowane godziny słuchania
        hour_counts = defaultdict(float)
        for features in self.artist_features.values():
            for hour, duration in features['listening_hours'].items():
                hour_counts[hour] += duration
        
        self.preferred_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)

        # Wykonaj klasteryzację artystów
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
    
    def analyze_taste_patterns(self) -> Dict:
        """
        Analizuje wzorce gustów muzycznych i tworzy profile słuchacza
        """
        print("\nAnalyzing taste patterns...")
        
        patterns = {
            'artist_clusters': {},  # Grupowanie artystów
            'listening_personas': [],  # Profile słuchacza
            'taste_vectors': {},  # Wektory preferencji
            'discovery_paths': []  # Ścieżki odkrywania muzyki
        }
        
        # Przygotuj dane o słuchaniu
        artist_features = defaultdict(lambda: {
            'play_count': 0,
            'total_time': 0,
            'time_of_day': defaultdict(int),
            'tags': set(),
            'first_listen': None,
            'last_listen': None,
            'listening_gaps': []
        })
        
        # Zbierz dane o artystach
        for _, row in tqdm(self.history_data.iterrows(), desc="Processing listening history"):
            artist = row['master_metadata_album_artist_name']
            timestamp = pd.to_datetime(row['ts'])
            ms_played = row['ms_played']
            
            # Aktualizuj statystyki artysty
            artist_features[artist]['play_count'] += 1
            artist_features[artist]['total_time'] += ms_played
            artist_features[artist]['time_of_day'][timestamp.hour] += 1
            
            # Śledź pierwszego i ostatniego odsłuchania
            if artist_features[artist]['first_listen'] is None:
                artist_features[artist]['first_listen'] = timestamp
            else:
                gap = (timestamp - artist_features[artist]['last_listen']).days
                if gap > 0:
                    artist_features[artist]['listening_gaps'].append(gap)
            artist_features[artist]['last_listen'] = timestamp

        # Utwórz klastry artystów
        print("\nClustering artists...")
        artist_vectors = []
        artist_names = []
        
        for artist, features in artist_features.items():
            if features['play_count'] >= 5:  # Minimum 5 odtworzeń
                vector = [
                    features['play_count'],
                    features['total_time'] / (1000 * 60),  # minuty
                    np.mean(features['listening_gaps']) if features['listening_gaps'] else 0,
                    np.std(features['listening_gaps']) if features['listening_gaps'] else 0,
                    *[features['time_of_day'][h] for h in range(24)]  # rozkład godzinowy
                ]
                artist_vectors.append(vector)
                artist_names.append(artist)
        
        if artist_vectors:
            # Normalizacja wektorów
            scaler = StandardScaler()
            artist_vectors_scaled = scaler.fit_transform(artist_vectors)
            
            # Klasteryzacja
            n_clusters = min(len(artist_vectors) // 5, 10)  # Max 10 klastrów
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(artist_vectors_scaled)
                
                # Zapisz klastry
                for artist, cluster in zip(artist_names, clusters):
                    patterns['artist_clusters'][artist] = int(cluster)
                
                # Analizuj profile słuchacza (persony)
                for cluster_id in range(n_clusters):
                    cluster_artists = [artist for i, artist in enumerate(artist_names) 
                                     if clusters[i] == cluster_id]
                    
                    # Zbierz cechy klastra
                    cluster_features = {
                        'artists': cluster_artists,
                        'size': len(cluster_artists),
                        'avg_plays': np.mean([artist_features[a]['play_count'] 
                                            for a in cluster_artists]),
                        'peak_hours': self._get_peak_hours([artist_features[a]['time_of_day'] 
                                                          for a in cluster_artists]),
                        'loyalty_score': np.mean([
                            len(artist_features[a]['listening_gaps']) /
                            (artist_features[a]['last_listen'] - 
                             artist_features[a]['first_listen']).days
                            for a in cluster_artists
                            if (artist_features[a]['last_listen'] - 
                                artist_features[a]['first_listen']).days > 0
                        ])
                    }
                    
                    patterns['listening_personas'].append(cluster_features)
        
        # Analiza ścieżek odkrywania
        print("\nAnalyzing discovery paths...")
        sorted_history = self.history_data.sort_values('ts')
        discovery_window = pd.Timedelta(days=7)
        current_path = []
        
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            if not current_path:
                current_path.append((artist, row['ts']))
            else:
                time_diff = row['ts'] - current_path[-1][1]
                if time_diff <= discovery_window and artist not in [a for a, _ in current_path]:
                    current_path.append((artist, row['ts']))
                elif time_diff > discovery_window:
                    if len(current_path) >= 3:  # Minimum 3 artystów w ścieżce
                        patterns['discovery_paths'].append([a for a, _ in current_path])
                    current_path = [(artist, row['ts'])]
        
        return patterns

    def _get_peak_hours(self, time_distributions: List[Dict]) -> List[int]:
        """Znajduje szczytowe godziny słuchania dla zbioru artystów"""
        combined = defaultdict(int)
        for dist in time_distributions:
            for hour, count in dist.items():
                combined[hour] += count
        
        # Zwróć top 3 godziny
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]

    def get_recommendations(self, top_n: int = 10) -> List[Dict]:
        """
        Generuje rekomendacje bazując na analizie wzorców
        """
        # Najpierw analizujemy wzorce
        patterns = self.analyze_taste_patterns()
        
        recommendations = []
        
        if not patterns['artist_clusters']:
            return self._get_basic_recommendations(top_n)
        
        # Znajdź aktywny profil słuchacza
        recent_artists = self.history_data.sort_values('ts').tail(50)[
            'master_metadata_album_artist_name'].unique()
        
        active_clusters = [patterns['artist_clusters'][artist] 
                          for artist in recent_artists 
                          if artist in patterns['artist_clusters']]
        
        if not active_clusters:
            return self._get_basic_recommendations(top_n)
        
        # Użyj najczęstszego klastra jako aktywnego profilu
        active_cluster = max(set(active_clusters), key=active_clusters.count)
        active_persona = patterns['listening_personas'][active_cluster]
        
        # Generuj rekomendacje bazując na aktywnym profilu
        for artist in active_persona['artists']:
            try:
                similar_artists = self.network.get_artist(artist).get_similar()
                for similar, match in similar_artists:
                    if str(similar) not in recent_artists:
                        recommendations.append({
                            'artist': str(similar),
                            'score': float(match) * active_persona['loyalty_score'],
                            'source': artist,
                            'cluster': active_cluster
                        })
            except Exception:
                continue
        
        # Sortuj i zwróć top_n rekomendacji
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

    def _get_basic_recommendations(self, top_n: int = 10) -> List[Dict]:
        """Podstawowe rekomendacje gdy brak wystarczających danych do analizy wzorców"""
        # ... (poprzedni kod rekomendacji)

    def _analyze_discovery_paths(self, sorted_history: pd.DataFrame, window_days: int = 7) -> List[List[str]]:
        """
        Analizuje ścieżki odkrywania nowych artystów w bardziej wydajny sposób.
        
        Args:
            sorted_history: Posortowana historia słuchania
            window_days: Okno czasowe w dniach dla jednej ścieżki odkrywania
        """
        print("\nAnalyzing discovery paths...")
        discovery_paths = []
        
        # Grupuj po dniach dla szybszej analizy
        daily_artists = sorted_history.groupby(sorted_history['ts'].dt.date).agg({
            'master_metadata_album_artist_name': lambda x: list(dict.fromkeys(x))  # zachowuje kolejność
        })
        
        window_size = pd.Timedelta(days=window_days)
        current_path = []
        current_artists = set()
        start_date = None
        
        for date, artists in tqdm(daily_artists.iterrows(), 
                                desc="Analyzing artist discovery patterns",
                                total=len(daily_artists)):
            if not current_path:
                start_date = date
                current_path.extend(artists['master_metadata_album_artist_name'])
                current_artists.update(current_path)
            else:
                if (date - start_date) <= window_size:
                    # Dodaj tylko nowych artystów z tego dnia
                    new_artists = [a for a in artists['master_metadata_album_artist_name'] 
                                 if a not in current_artists]
                    if new_artists:
                        current_path.extend(new_artists)
                        current_artists.update(new_artists)
                else:
                    # Zapisz ścieżkę jeśli jest wystarczająco długa
                    if len(current_path) >= 3:
                        discovery_paths.append(current_path)
                    # Zacznij nową ścieżkę
                    current_path = artists['master_metadata_album_artist_name']
                    current_artists = set(current_path)
                    start_date = date
        
        # Dodaj ostatnią ścieżkę jeśli spełnia kryteria
        if len(current_path) >= 3:
            discovery_paths.append(current_path)
        
        print(f"Found {len(discovery_paths)} discovery paths")
        return discovery_paths

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