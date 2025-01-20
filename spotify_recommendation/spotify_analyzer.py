import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from typing import Dict, List, Set, Tuple
import warnings
from datetime import datetime
import calendar

warnings.filterwarnings('ignore')
load_dotenv()

class SpotifyAnalyzer:
    def __init__(self, data_dir: str = 'data'):
        """
        Inicjalizacja analizatora i rekomendatora muzyki.
        """
        # Podstawowe atrybuty
        self.history_data = None
        self.spotify = None
        
        # Cechy artystów i utworów
        self.artist_features = defaultdict(lambda: {
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
        })
        
        # Wyniki analiz
        self.favorite_genres = None
        self.preferred_hours = None
        self.artist_clusters = None
        self.listening_patterns = None
        self.genre_patterns = None
        self.temporal_patterns = None
        
        # Cache
        self.cache_dir = Path(data_dir) / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Inicjalizacja
        self.load_streaming_history(data_dir)
        self.setup_spotify_client()
        
    def setup_spotify_client(self):
        """Inicjalizacja klienta Spotify API"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                raise ValueError("Missing Spotify credentials. Check your .env file")
            
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            print("Successfully connected to Spotify API!")
        except Exception as e:
            print(f"Error setting up Spotify client: {e}")
            self.spotify = None

    def load_streaming_history(self, data_dir: str) -> None:
        """Wczytuje historię słuchania z plików Spotify"""
        all_data = []
        data_path = Path(data_dir)
        
        for file in data_path.glob('Streaming_History_Audio_*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"No streaming history found in {data_dir}")
        
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

    def process_artist_data(self):
        """Przetwarza dane artystów z wykorzystaniem cache'u"""
        print("\nProcessing artist data...")
        
        # Wczytaj cache
        cache_file = self.cache_dir / 'artist_data.json'
        artist_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    artist_cache = json.load(f)
                print(f"Loaded {len(artist_cache)} artists from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Zbierz podstawowe statystyki
        for _, row in tqdm(self.history_data.iterrows(), desc="Processing listening history"):
            artist = row['master_metadata_album_artist_name']
            track = row['master_metadata_track_name']
            ms_played = row['ms_played']
            timestamp = pd.to_datetime(row['ts'])
            
            features = self.artist_features[artist]
            features['play_count'] += 1
            features['total_ms'] += ms_played
            features['unique_tracks'].add(track)
            features['listening_hours'][timestamp.hour] += ms_played / (1000 * 60 * 60)
            
            # Śledzenie pierwszego i ostatniego odsłuchania
            if features['first_listen'] is None:
                features['first_listen'] = timestamp
            else:
                gap = (timestamp - pd.to_datetime(features['last_listen'])).days
                if gap > 0:
                    features['listening_gaps'].append(gap)
            features['last_listen'] = timestamp.isoformat()

        # Pobierz dane dla nowych artystów
        uncached_artists = [artist for artist in self.artist_features 
                           if artist not in artist_cache]
        
        if uncached_artists and self.spotify:
            print(f"\nFetching data for {len(uncached_artists)} new artists...")
            
            for artist_name in tqdm(uncached_artists, desc="Fetching Spotify data"):
                try:
                    results = self.spotify.search(q=artist_name, type='artist', limit=1)
                    if results['artists']['items']:
                        artist = results['artists']['items'][0]
                        
                        artist_cache[artist_name] = {
                            'genres': artist['genres'],
                            'popularity': artist['popularity'],
                            'spotify_id': artist['id']
                        }
                        
                        # Aktualizuj features
                        self.artist_features[artist_name].update({
                            'genres': set(artist['genres']),
                            'global_popularity': artist['popularity']
                        })
                        
                except Exception as e:
                    print(f"Error processing {artist_name}: {e}")
                    continue
            
            # Zapisz cache
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(artist_cache, f, ensure_ascii=False, indent=2)
                print(f"Updated cache with {len(artist_cache)} artists")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        # Użyj danych z cache'u
        for artist_name, cache_data in artist_cache.items():
            if artist_name in self.artist_features:
                self.artist_features[artist_name].update({
                    'genres': set(cache_data['genres']),
                    'global_popularity': cache_data['popularity']
                })

    def analyze_all(self) -> Dict:
        """Przeprowadza wszystkie analizy i zwraca kompleksowy raport"""
        print("\nStarting comprehensive music analysis...")
        
        # Najpierw przetwórz dane artystów
        self.process_artist_data()
        
        analysis = {
            'temporal': self.analyze_temporal_patterns(),
            'genres': self.analyze_genre_patterns(),
            'listening': self.analyze_listening_patterns(),
            'discovery': self.analyze_discovery_paths(),
            'recommendations': self.get_recommendations()
        }
        
        return analysis

    def analyze_temporal_patterns(self) -> Dict:
        """Analizuje wzorce czasowe słuchania muzyki"""
        print("\nAnalyzing temporal patterns...")
        
        patterns = {
            'seasonal': defaultdict(int),
            'monthly': defaultdict(int),
            'daily': defaultdict(int),
            'weekday': defaultdict(int),
            'session_length': [],
            'listening_evolution': [],
            'peak_hours': {},
            'artist_retention': {}
        }
        
        # Analiza sezonowa i miesięczna
        for _, row in tqdm(self.history_data.iterrows(), desc="Processing temporal data"):
            ts = row['ts']
            ms_played = row['ms_played']
            
            if ms_played < 30000:  # Pomijamy bardzo krótkie odtworzenia
                continue
            
            # Sezon
            season = (ts.month % 12 + 3) // 3
            patterns['seasonal'][f"Q{season}"] += ms_played
            
            # Miesiąc
            patterns['monthly'][ts.strftime('%Y-%m')] += ms_played
            
            # Dzień tygodnia
            patterns['weekday'][calendar.day_name[ts.weekday()]] += ms_played
            
            # Godzina
            patterns['daily'][ts.hour] += ms_played

        # Analiza sesji słuchania
        sorted_history = self.history_data.sort_values('ts')
        session_threshold = pd.Timedelta(minutes=30)
        current_session = []
        
        for i in range(len(sorted_history)-1):
            current = sorted_history.iloc[i]
            next_track = sorted_history.iloc[i+1]
            time_diff = next_track['ts'] - current['ts']
            
            current_session.append(current)
            if time_diff > session_threshold:
                if len(current_session) >= 3:
                    session_length = sum(track['ms_played'] for track in current_session) / (1000 * 60)
                    patterns['session_length'].append(session_length)
                current_session = []

        # Analiza ewolucji słuchania
        monthly_stats = sorted_history.groupby(pd.Grouper(key='ts', freq='M')).agg({
            'master_metadata_album_artist_name': 'nunique',
            'ms_played': 'sum'
        })
        
        patterns['listening_evolution'] = [{
            'month': month.strftime('%Y-%m'),
            'unique_artists': row['master_metadata_album_artist_name'],
            'total_hours': row['ms_played'] / (1000 * 60 * 60)
        } for month, row in monthly_stats.iterrows()]

        # Analiza retencji artystów
        for artist, features in self.artist_features.items():
            if features['listening_gaps']:
                patterns['artist_retention'][artist] = {
                    'avg_gap': np.mean(features['listening_gaps']),
                    'max_gap': max(features['listening_gaps']),
                    'consistency': len(features['listening_gaps']) / 
                                 ((pd.to_datetime(features['last_listen']) - 
                                   features['first_listen']).days + 1)
                }

        return patterns

    def analyze_genre_patterns(self) -> Dict:
        """Analizuje wzorce gatunkowe i ich ewolucję"""
        print("\nAnalyzing genre patterns...")
        
        patterns = {
            'genre_evolution': defaultdict(list),
            'genre_combinations': defaultdict(int),
            'genre_flow': defaultdict(lambda: defaultdict(int)),
            'genre_expertise': defaultdict(float),
            'genre_exploration': []
        }
        
        # Analiza ewolucji gatunków w czasie
        monthly_data = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        
        for month, group in tqdm(monthly_data, desc="Processing genre evolution"):
            month_genres = defaultdict(int)
            for _, row in group.iterrows():
                artist = row['master_metadata_album_artist_name']
                if artist in self.artist_features:
                    for genre in self.artist_features[artist]['genres']:
                        month_genres[genre] += row['ms_played']
            
            # Normalizacja i zapis
            total_time = sum(month_genres.values())
            if total_time > 0:
                for genre, time in month_genres.items():
                    patterns['genre_evolution'][genre].append({
                        'month': month.strftime('%Y-%m'),
                        'share': time/total_time
                    })

        # Analiza kombinacji gatunków
        for artist, features in self.artist_features.items():
            genres = list(features['genres'])
            for i in range(len(genres)):
                for j in range(i+1, len(genres)):
                    pair = tuple(sorted([genres[i], genres[j]]))
                    patterns['genre_combinations'][pair] += features['play_count']

        # Analiza przepływów między gatunkami
        sorted_history = self.history_data.sort_values('ts')
        prev_genres = None
        
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            if artist in self.artist_features:
                current_genres = self.artist_features[artist]['genres']
                if prev_genres:
                    for prev_genre in prev_genres:
                        for curr_genre in current_genres:
                            if prev_genre != curr_genre:
                                patterns['genre_flow'][prev_genre][curr_genre] += 1
                prev_genres = current_genres

        # Obliczanie ekspertyzy w gatunkach
        total_time = sum(f['total_ms'] for f in self.artist_features.values())
        if total_time > 0:
            for artist, features in self.artist_features.items():
                weight = features['total_ms'] / total_time
                for genre in features['genres']:
                    patterns['genre_expertise'][genre] += weight

        return patterns 

    def analyze_listening_patterns(self) -> Dict:
        """Analizuje szczegółowe wzorce słuchania"""
        print("\nAnalyzing listening patterns...")
        
        patterns = {
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
        
        # Analiza ulubionych artystów w czasie
        monthly_top = self.history_data.groupby([
            pd.Grouper(key='ts', freq='M'),
            'master_metadata_album_artist_name'
        ]).size().reset_index(name='plays')
        
        for _, month_data in monthly_top.groupby(pd.Grouper(key='ts', freq='M')):
            top_artists = month_data.nlargest(5, 'plays')
            patterns['listening_habits']['favorite_artists'].append({
                'month': month_data['ts'].iloc[0].strftime('%Y-%m'),
                'artists': [{
                    'name': row['master_metadata_album_artist_name'],
                    'plays': row['plays']
                } for _, row in top_artists.iterrows()]
            })
        
        # Analiza odkrywania nowych artystów
        known_artists = set()
        monthly_discovery = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        
        for month, group in monthly_discovery:
            month_artists = set(group['master_metadata_album_artist_name'])
            new_artists = month_artists - known_artists
            known_artists.update(new_artists)
            
            patterns['listening_habits']['discovery_rate'].append({
                'month': month.strftime('%Y-%m'),
                'new_artists': len(new_artists),
                'total_artists': len(month_artists),
                'discovery_rate': len(new_artists) / len(month_artists) if month_artists else 0
            })
        
        # Analiza powtórzeń
        for artist, features in self.artist_features.items():
            patterns['listening_habits']['repeat_patterns'][features['play_count']] += 1
        
        # Analiza "streaks" słuchania
        current_streak = []
        prev_date = None
        
        for date in sorted(self.history_data['ts'].dt.date.unique()):
            if prev_date is None or (date - prev_date).days == 1:
                current_streak.append(date)
            else:
                if len(current_streak) >= 3:
                    patterns['listening_habits']['listening_streaks'].append({
                        'start': current_streak[0].isoformat(),
                        'end': current_streak[-1].isoformat(),
                        'length': len(current_streak)
                    })
                current_streak = [date]
            prev_date = date
        
        return patterns

    def analyze_discovery_paths(self) -> Dict:
        """Analizuje ścieżki odkrywania nowej muzyki"""
        print("\nAnalyzing discovery patterns...")
        
        paths = {
            'artist_paths': [],
            'genre_paths': [],
            'discovery_sources': defaultdict(int),
            'exploration_patterns': []
        }
        
        # Znajdź ścieżki między artystami
        sorted_history = self.history_data.sort_values('ts')
        window_size = pd.Timedelta(days=7)
        current_path = []
        current_artists = set()
        start_time = None
        
        for _, row in sorted_history.iterrows():
            artist = row['master_metadata_album_artist_name']
            if not current_path:
                start_time = row['ts']
                current_path.append(artist)
                current_artists.add(artist)
            else:
                if (row['ts'] - start_time) <= window_size and artist not in current_artists:
                    # Sprawdź podobieństwo gatunków z poprzednimi artystami
                    is_related = False
                    if artist in self.artist_features:
                        for prev_artist in current_path[-3:]:  # sprawdź ostatnich 3 artystów
                            if prev_artist in self.artist_features:
                                common_genres = (self.artist_features[artist]['genres'] & 
                                              self.artist_features[prev_artist]['genres'])
                                if len(common_genres) >= 2:  # minimum 2 wspólne gatunki
                                    is_related = True
                                    paths['discovery_sources']['genre_related'] += 1
                                    break
                    
                    current_path.append(artist)
                    current_artists.add(artist)
                    
                    if not is_related:
                        paths['discovery_sources']['new_discovery'] += 1
                
                elif (row['ts'] - start_time) > window_size:
                    if len(current_path) >= 3:
                        paths['artist_paths'].append({
                            'path': current_path,
                            'start_time': start_time.isoformat(),
                            'duration_hours': (row['ts'] - start_time).total_seconds() / 3600
                        })
                    current_path = [artist]
                    current_artists = {artist}
                    start_time = row['ts']
        
        # Analiza eksploracji gatunków
        known_genres = set()
        monthly_genres = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))
        
        for month, group in monthly_genres:
            month_genres = set()
            for _, row in group.iterrows():
                artist = row['master_metadata_album_artist_name']
                if artist in self.artist_features:
                    month_genres.update(self.artist_features[artist]['genres'])
            
            new_genres = month_genres - known_genres
            known_genres.update(new_genres)
            
            paths['exploration_patterns'].append({
                'month': month.strftime('%Y-%m'),
                'new_genres': list(new_genres),
                'total_genres': len(month_genres),
                'exploration_rate': len(new_genres) / len(month_genres) if month_genres else 0
            })
        
        return paths

    def get_recommendations(self, top_n: int = 10) -> Dict[str, List[Dict]]:
        """Generuje spersonalizowane rekomendacje bazując na podobieństwie gatunków"""
        print("\nGenerating recommendations...")
        
        recommendations = {
            'mainstream': [],
            'hidden_gems': []
        }
        
        recent_artists = set(self.history_data.nlargest(100, 'ts')['master_metadata_album_artist_name'])
        
        # Znajdź aktywne gatunki
        recent_genres = set()
        for artist in recent_artists:
            if artist in self.artist_features:
                recent_genres.update(self.artist_features[artist]['genres'])
        
        # Znajdź podobnych artystów bazując na gatunkach
        all_artists = set(self.artist_features.keys()) - recent_artists
        potential_recommendations = []
        
        for artist in tqdm(all_artists, desc="Finding similar artists"):
            if not self.artist_features[artist]['genres']:
                continue
            
            # Oblicz podobieństwo gatunków
            artist_genres = self.artist_features[artist]['genres']
            genre_match = len(artist_genres & recent_genres)
            genre_diversity = len(artist_genres)
            popularity = self.artist_features[artist]['global_popularity'] / 100
            
            # Oblicz score bazując na różnych czynnikach
            genre_score = (0.6 * genre_match / max(1, len(recent_genres)) + 
                          0.4 * genre_diversity / max(1, len(artist_genres)))
            
            if genre_score > 0.3:  # tylko znaczące dopasowania gatunkowe
                potential_recommendations.append({
                    'artist': artist,
                    'genres': list(artist_genres),
                    'popularity': self.artist_features[artist]['global_popularity'],
                    'genre_score': genre_score,
                    'popularity_score': popularity,
                    'total_score': 0.7 * genre_score + 0.3 * popularity
                })
        
        # Sortuj według całkowitego score
        sorted_recommendations = sorted(
            potential_recommendations, 
            key=lambda x: x['total_score'], 
            reverse=True
        )
        
        # Podziel na mainstream i hidden gems
        for rec in sorted_recommendations:
            if len(recommendations['mainstream']) < top_n and rec['popularity'] >= 70:
                recommendations['mainstream'].append(rec)
            elif len(recommendations['hidden_gems']) < top_n and rec['popularity'] < 70 and rec['genre_score'] > 0.4:
                recommendations['hidden_gems'].append(rec)
            
            if len(recommendations['mainstream']) >= top_n and len(recommendations['hidden_gems']) >= top_n:
                break
        
        return recommendations

    def _cluster_artists(self) -> Dict:
        """Grupuje artystów w klastry na podstawie cech"""
        print("\nClustering artists...")
        
        # Przygotuj dane do klasteryzacji
        features = []
        artists = []
        
        for artist, data in self.artist_features.items():
            if data['play_count'] >= 5:  # minimum 5 odtworzeń
                feature_vector = [
                    data['play_count'],
                    data['total_ms'] / (1000 * 60 * 60),  # godziny
                    len(data['unique_tracks']),
                    data['global_popularity'],
                    len(data['genres']),
                    np.mean(data['listening_gaps']) if data['listening_gaps'] else 0
                ]
                features.append(feature_vector)
                artists.append(artist)
        
        if not features:
            return {}
        
        # Normalizacja
        features_scaled = StandardScaler().fit_transform(features)
        
        # Znajdź optymalną liczbę klastrów
        max_clusters = min(len(features) // 5, 10)  # max 10 klastrów
        if max_clusters < 2:
            return {}
        
        kmeans = KMeans(n_clusters=max_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analizuj klastry
        cluster_data = defaultdict(lambda: {
            'artists': [],
            'genres': defaultdict(int),
            'avg_popularity': 0,
            'avg_plays': 0,
            'size': 0
        })
        
        for artist, cluster in zip(artists, clusters):
            data = self.artist_features[artist]
            cluster_info = cluster_data[int(cluster)]
            cluster_info['artists'].append(artist)
            cluster_info['avg_popularity'] += data['global_popularity']
            cluster_info['avg_plays'] += data['play_count']
            cluster_info['size'] += 1
            for genre in data['genres']:
                cluster_info['genres'][genre] += 1
        
        # Normalizuj statystyki klastrów
        for cluster_info in cluster_data.values():
            if cluster_info['size'] > 0:
                cluster_info['avg_popularity'] /= cluster_info['size']
                cluster_info['avg_plays'] /= cluster_info['size']
                cluster_info['top_genres'] = sorted(
                    cluster_info['genres'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
        
        return dict(cluster_data)

    def _analyze_listening_moods(self) -> Dict:
        """Analizuje nastroje słuchania w różnych kontekstach"""
        moods = {
            'time_of_day': defaultdict(lambda: defaultdict(int)),
            'weekday': defaultdict(lambda: defaultdict(int)),
            'seasonal': defaultdict(lambda: defaultdict(int))
        }
        
        for _, row in self.history_data.iterrows():
            artist = row['master_metadata_album_artist_name']
            if artist not in self.artist_features:
                continue
            
            ts = row['ts']
            hour = ts.hour
            weekday = ts.strftime('%A')
            season = (ts.month % 12 + 3) // 3
            
            for genre in self.artist_features[artist]['genres']:
                moods['time_of_day'][hour][genre] += 1
                moods['weekday'][weekday][genre] += 1
                moods['seasonal'][f"Q{season}"][genre] += 1
        
        return moods

    def _analyze_artist_relationships(self) -> Dict:
        """Analizuje relacje między artystami"""
        relationships = {
            'genre_similarity': defaultdict(float),
            'listening_patterns': defaultdict(lambda: defaultdict(int)),
            'co_occurrence': defaultdict(lambda: defaultdict(int))
        }
        
        # Znajdź artystów często słuchanych razem
        session_threshold = pd.Timedelta(minutes=30)
        current_session = []
        
        sorted_history = self.history_data.sort_values('ts')
        for i in range(len(sorted_history)-1):
            current = sorted_history.iloc[i]
            next_track = sorted_history.iloc[i+1]
            
            if (next_track['ts'] - current['ts']) <= session_threshold:
                current_session.append(current['master_metadata_album_artist_name'])
            else:
                if len(current_session) >= 2:
                    for j in range(len(current_session)-1):
                        for k in range(j+1, len(current_session)):
                            artist1, artist2 = sorted([current_session[j], current_session[k]])
                            relationships['listening_patterns'][artist1][artist2] += 1
                current_session = [current['master_metadata_album_artist_name']]
        
        # Oblicz podobieństwo gatunków
        artists = list(self.artist_features.keys())
        for i in range(len(artists)):
            for j in range(i+1, len(artists)):
                artist1, artist2 = artists[i], artists[j]
                genres1 = self.artist_features[artist1]['genres']
                genres2 = self.artist_features[artist2]['genres']
                
                if genres1 and genres2:
                    similarity = len(genres1 & genres2) / len(genres1 | genres2)
                    if similarity > 0.3:  # tylko znaczące podobieństwa
                        relationships['genre_similarity'][artist1, artist2] = similarity
        
        # Analizuj współwystępowanie w czasie
        monthly_artists = self.history_data.groupby(pd.Grouper(key='ts', freq='M'))[
            'master_metadata_album_artist_name'].unique()
        
        for month_artists in monthly_artists:
            for i in range(len(month_artists)):
                for j in range(i+1, len(month_artists)):
                    artist1, artist2 = sorted([month_artists[i], month_artists[j]])
                    relationships['co_occurrence'][artist1][artist2] += 1
        
        return relationships

if __name__ == "__main__":
    analyzer = SpotifyAnalyzer()
    
    print("\nRunning comprehensive music analysis...")
    analysis = analyzer.analyze_all()
    
    print("\n=== Analysis Results ===")
    
    # Statystyki czasowe
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
        for season, count in sorted(temp['seasonal'].items(), key=lambda x: x[1], reverse=True):
            print(f"- {season}: {count/sum(temp['seasonal'].values()):.1%}")
        
        if temp['session_length']:
            print(f"\nListening sessions:")
            print(f"- Average length: {np.mean(temp['session_length']):.1f} minutes")
            print(f"- Longest session: {np.max(temp['session_length']):.1f} minutes")
            print(f"- Total sessions: {len(temp['session_length'])}")
    
    # Gatunki i ich ewolucja
    print("\nGenre Analysis:")
    if 'genres' in analysis:
        genres = analysis['genres']
        
        print("\nTop genres by expertise:")
        for genre, score in sorted(genres['genre_expertise'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"- {genre}: {score:.1%}")
        
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
    
    # Wzorce odkrywania
    print("\nDiscovery Patterns:")
    if 'discovery' in analysis:
        discovery = analysis['discovery']
        
        print("\nDiscovery sources:")
        total_discoveries = sum(discovery['discovery_sources'].values())
        for source, count in sorted(discovery['discovery_sources'].items(), key=lambda x: x[1], reverse=True):
            print(f"- {source}: {count} ({count/total_discoveries:.1%})")
        
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
    
    # Wzorce słuchania
    print("\nListening Patterns:")
    if 'listening' in analysis:
        listening = analysis['listening']
        
        if 'listening_habits' in listening and listening['listening_habits']['favorite_artists']:
            recent_favorites = listening['listening_habits']['favorite_artists'][-1]
            print(f"\nTop artists ({recent_favorites['month']}):")
            for artist in recent_favorites['artists']:
                print(f"- {artist['name']}: {artist['plays']} plays")
        
        if 'artist_clusters' in listening:
            print("\nArtist clusters:")
            for cluster_id, cluster in listening['artist_clusters'].items():
                if isinstance(cluster, dict) and cluster.get('size', 0) >= 5:
                    print(f"\nCluster {cluster_id} ({cluster['size']} artists):")
                    print(f"- Average popularity: {cluster['avg_popularity']:.1f}")
                    if 'top_genres' in cluster:
                        print(f"- Top genres: {', '.join(genre for genre, _ in cluster['top_genres'][:3])}")
                    print(f"- Example artists: {', '.join(cluster['artists'][:3])}")
    
    # Rekomendacje
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
    
    print("\n=== Analysis Complete ===") 