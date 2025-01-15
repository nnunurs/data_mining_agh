import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import calendar
from typing import Dict, List, Set
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from textblob import TextBlob
import librosa
import warnings
import json
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')
import os

class MusicAnalyzer:
    def __init__(self, data_dir: str = 'data'):
        """
        Inicjalizacja analizatora muzycznego.
        
        Args:
            data_dir: Ścieżka do folderu z plikami historii Spotify
        """
        self.history_data = None
        self.spotify = None
        self.load_streaming_history(data_dir)
        self.setup_spotify_client()

    def setup_spotify_client(self):
        """Inicjalizacja klienta Spotify API"""
        try:
            client_credentials_manager = SpotifyClientCredentials(client_id=os.getenv('SPOTIFY_CLIENT_ID'), client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'))
            self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            print(f"Error setting up Spotify client: {e}")
            self.spotify = None

    def load_streaming_history(self, data_dir: str) -> None:
        """
        Wczytuje historię słuchania z plików Spotify.
        
        Args:
            data_dir: Ścieżka do folderu z plikami historii
        """
        all_data = []
        data_path = Path(data_dir)
        
        # Wczytaj wszystkie pliki historii Spotify
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
        
        # Konwersja do DataFrame
        self.history_data = pd.DataFrame(all_data)
        
        # Filtrowanie niepełnych danych
        self.history_data = self.history_data[
            self.history_data['master_metadata_track_name'].notna() &
            self.history_data['master_metadata_album_artist_name'].notna()
        ]
        
        # Konwersja timestamp
        self.history_data['ts'] = pd.to_datetime(self.history_data['ts'])
        
        print(f"\nLoaded streaming history:")
        print(f"Total tracks: {len(self.history_data)}")
        print(f"Unique artists: {self.history_data['master_metadata_album_artist_name'].nunique()}")
        print(f"Date range: {self.history_data['ts'].min()} to {self.history_data['ts'].max()}")

    def analyze_temporal_patterns(self) -> Dict:
        """Analiza wzorców czasowych słuchania muzyki"""
        if self.history_data is None:
            raise ValueError("No listening history data available")

        patterns = {
            'seasonal': defaultdict(int),
            'monthly': defaultdict(int),
            'daily': defaultdict(int),
            'weekday': defaultdict(int),
            'session_length': [],
            'genre_time': defaultdict(lambda: defaultdict(int))
        }

        # Konwertuj timestamp na datetime jeśli nie jest
        if not pd.api.types.is_datetime64_any_dtype(self.history_data['ts']):
            self.history_data['ts'] = pd.to_datetime(self.history_data['ts'])

        # Analiza sezonowa i miesięczna
        for _, row in self.history_data.iterrows():
            ts = row['ts']
            ms_played = row['ms_played']
            
            # Pomijamy bardzo krótkie odtworzenia
            if ms_played < 30000:  # mniej niż 30 sekund
                continue

            # Sezon
            season = (ts.month % 12 + 3) // 3
            patterns['seasonal'][f"Q{season}"] += ms_played

            # Miesiąc
            patterns['monthly'][calendar.month_name[ts.month]] += ms_played

            # Dzień tygodnia
            patterns['weekday'][calendar.day_name[ts.weekday()]] += ms_played

            # Godzina
            patterns['daily'][ts.hour] += ms_played

        # Analiza sesji słuchania
        sorted_history = self.history_data.sort_values('ts')
        session_threshold = pd.Timedelta(minutes=30)
        current_session = 0
        session_start = None

        for idx, row in sorted_history.iterrows():
            if session_start is None:
                session_start = row['ts']
                current_session = row['ms_played']
            else:
                time_diff = row['ts'] - session_start
                if time_diff <= session_threshold:
                    current_session += row['ms_played']
                else:
                    patterns['session_length'].append(current_session / (1000 * 60))  # konwersja na minuty
                    session_start = row['ts']
                    current_session = row['ms_played']

        # Normalizacja wartości
        total_time = sum(patterns['daily'].values())
        for key in ['seasonal', 'monthly', 'daily', 'weekday']:
            patterns[key] = {k: v/total_time for k, v in patterns[key].items()}

        return patterns

    def analyze_mood_patterns(self) -> Dict:
        """Analiza nastrojowa muzyki bazująca na metadanych"""
        mood_analysis = {
            'time_of_day': defaultdict(int),
            'track_length_distribution': [],
            'favorite_periods': defaultdict(int)
        }

        for _, row in self.history_data.iterrows():
            hour = pd.to_datetime(row['ts']).hour
            ms_played = row['ms_played']
            
            if ms_played < 30000:  # pomijamy krótkie odtworzenia
                continue
            
            # Analiza pór dnia
            if 6 <= hour < 12:
                period = 'morning'
            elif 12 <= hour < 17:
                period = 'afternoon'
            elif 17 <= hour < 22:
                period = 'evening'
            else:
                period = 'night'
            
            mood_analysis['time_of_day'][hour] += ms_played
            mood_analysis['favorite_periods'][period] += ms_played
            mood_analysis['track_length_distribution'].append(ms_played / 1000)  # konwersja na sekundy

        # Normalizacja
        total_time = sum(mood_analysis['time_of_day'].values())
        mood_analysis['time_of_day'] = {k: v/total_time for k, v in mood_analysis['time_of_day'].items()}
        mood_analysis['favorite_periods'] = {k: v/total_time for k, v in mood_analysis['favorite_periods'].items()}
        
        # Statystyki długości utworów
        mood_analysis['track_length_stats'] = {
            'mean': np.mean(mood_analysis['track_length_distribution']),
            'median': np.median(mood_analysis['track_length_distribution']),
            'std': np.std(mood_analysis['track_length_distribution'])
        }

        return mood_analysis

    def analyze_genre_depth(self) -> Dict:
        """Analiza głębokości znajomości gatunków"""
        if self.spotify is None:
            print("Warning: Spotify client not initialized, skipping genre analysis")
            return {
                'genre_progression': {},
                'subgenre_exploration': {},
                'genre_diversity': [],
                'genre_expertise': {},
                'genre_relationships': {}
            }

        print("\nAnalyzing genre patterns...")
        
        # Ścieżka do pliku cache
        cache_file = Path('data/genre_cache.json')
        cache_file.parent.mkdir(exist_ok=True)
        
        # Wczytaj cache jeśli istnieje
        artist_genres = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    artist_genres = json.load(f)
                print(f"Loaded {len(artist_genres)} artists from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        genre_analysis = {
            'genre_progression': defaultdict(list),
            'subgenre_exploration': defaultdict(set),
            'genre_diversity': [],
            'genre_expertise': defaultdict(float),
            'genre_relationships': defaultdict(lambda: defaultdict(int))
        }

        # Słownik do mapowania podgatunków na główne gatunki
        main_genres = {
            'progressive rock': 'rock', 'hard rock': 'rock', 'indie rock': 'rock',
            'deep house': 'electronic', 'techno': 'electronic', 'ambient': 'electronic',
            'bebop': 'jazz', 'fusion': 'jazz', 'swing': 'jazz',
        }

        # Zbierz dane o czasie słuchania dla każdego artysty
        artist_playtime = defaultdict(int)
        for _, row in self.history_data.iterrows():
            artist_name = row['master_metadata_album_artist_name']
            ms_played = row['ms_played']
            artist_playtime[artist_name] += ms_played

        # Znajdź artystów, których nie ma w cache
        uncached_artists = [artist for artist in artist_playtime.keys() 
                           if artist not in artist_genres]
        
        if uncached_artists:
            print(f"\nFetching genres for {len(uncached_artists)} new artists...")
            
            for artist_name in tqdm(uncached_artists, desc="Processing new artists"):
                try:
                    results = self.spotify.search(q=artist_name, type='artist', limit=1)
                    if results['artists']['items']:
                        genres = list(set(results['artists']['items'][0]['genres']))  # Konwersja do listy dla JSON
                        artist_genres[artist_name] = genres
                except Exception as e:
                    print(f"Error fetching genres for {artist_name}: {e}")
                    continue
            
            # Zapisz zaktualizowany cache
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(artist_genres, f, ensure_ascii=False, indent=2)
                print(f"Updated cache with {len(artist_genres)} artists")
            except Exception as e:
                print(f"Error saving cache: {e}")

        # Analiza chronologiczna po miesiącach
        print("\nAnalyzing genre progression...")
        monthly_data = self.history_data.groupby(self.history_data['ts'].dt.strftime('%Y-%m'))
        
        for month, group in tqdm(monthly_data, desc="Processing monthly data"):
            month_genres = set()
            for _, row in group.iterrows():
                artist = row['master_metadata_album_artist_name']
                if artist in artist_genres:
                    month_genres.update(artist_genres[artist])
            
            genre_analysis['genre_diversity'].append((month, len(month_genres)))

        # Analiza podgatunków i relacji
        print("\nAnalyzing genre relationships...")
        for artist, genres in artist_genres.items():
            if artist not in artist_playtime:  # Pomijamy artystów, których nie ma w aktualnej historii
                continue
            
            playtime = artist_playtime[artist]
            
            for genre in genres:
                # Mapowanie na główne gatunki
                main_genre = next((main for sub, main in main_genres.items() 
                                 if sub in genre), genre)
                genre_analysis['subgenre_exploration'][main_genre].add(genre)
                
                # Aktualizacja ekspertyzy
                genre_analysis['genre_expertise'][genre] += playtime
                
                # Relacje między gatunkami
                for other_genre in genres:
                    if other_genre != genre:
                        genre_analysis['genre_relationships'][genre][other_genre] += 1

        # Normalizacja ekspertyzy
        total_time = sum(genre_analysis['genre_expertise'].values())
        if total_time > 0:
            genre_analysis['genre_expertise'] = {
                genre: time/total_time 
                for genre, time in genre_analysis['genre_expertise'].items()
            }

        return genre_analysis

    def analyze_musical_features(self) -> Dict:
        """Analiza cech muzycznych bazująca na metadanych"""
        musical_analysis = {
            'artist_frequency': defaultdict(int),
            'track_frequency': defaultdict(int),
            'listening_patterns': {
                'total_time': 0,
                'average_track_length': 0,
                'tracks_per_session': [],
                'repeat_patterns': defaultdict(int)
            }
        }

        # Analiza częstotliwości
        for _, row in self.history_data.iterrows():
            artist = row['master_metadata_album_artist_name']
            track = row['master_metadata_track_name']
            ms_played = row['ms_played']
            
            if ms_played < 30000:
                continue
            
            musical_analysis['artist_frequency'][artist] += 1
            musical_analysis['track_frequency'][f"{artist} - {track}"] += 1
            musical_analysis['listening_patterns']['total_time'] += ms_played

        # Obliczanie statystyk
        total_tracks = sum(musical_analysis['track_frequency'].values())
        musical_analysis['listening_patterns']['average_track_length'] = \
            musical_analysis['listening_patterns']['total_time'] / (total_tracks * 1000)  # w sekundach

        # Analiza powtórzeń
        for track, plays in musical_analysis['track_frequency'].items():
            musical_analysis['repeat_patterns'][plays] += 1

        # Top artyści i utwory
        musical_analysis['top_artists'] = sorted(
            musical_analysis['artist_frequency'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        musical_analysis['top_tracks'] = sorted(
            musical_analysis['track_frequency'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return musical_analysis

    def get_comprehensive_analysis(self) -> Dict:
        """Przeprowadza wszystkie analizy i zwraca kompleksowy raport"""
        analysis_report = {
            'temporal': self.analyze_temporal_patterns(),
            'mood': self.analyze_mood_patterns(),
            'genre': self.analyze_genre_depth(),
            'musical': self.analyze_musical_features()
        }
        
        return analysis_report