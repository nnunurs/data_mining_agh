import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyClient:
    def __init__(self):
        self.spotify = self._setup_client()
    
    def _setup_client(self):
        """Initializes Spotify API client"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                raise ValueError("Missing Spotify credentials. Check your .env file")
            
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            print("Successfully connected to Spotify API!")
            return spotify
        except Exception as e:
            print(f"Error setting up Spotify client: {e}")
            return None 