import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from pathlib import Path
import pylast

# Mockujemy dotenv przed importem spotify_recommender
with patch('dotenv.load_dotenv', return_value=True):
    from spotify_recommender import MusicRecommender

class TestMusicRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path("test_data")
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create sample streaming history
        cls.sample_history = [
            {
                "ts": "2024-01-01T12:00:00Z",
                "master_metadata_album_artist_name": "Test Artist 1",
                "master_metadata_track_name": "Test Track 1",
                "ms_played": 240000
            },
            {
                "ts": "2024-01-01T13:00:00Z",
                "master_metadata_album_artist_name": "Test Artist 2",
                "master_metadata_track_name": "Test Track 2",
                "ms_played": 180000
            }
        ]
        
        with open(cls.test_data_dir / "Streaming_History_Audio_1.json", 'w') as f:
            json.dump(cls.sample_history, f)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.test_data_dir)

    def setUp(self):
        self.recommender = MusicRecommender()

    def test_load_streaming_history(self):
        history = self.recommender.load_streaming_history(self.test_data_dir)
        self.assertIsInstance(history, pd.DataFrame)
        self.assertEqual(len(history), 2)
        self.assertEqual(history['master_metadata_album_artist_name'].nunique(), 2)

    @patch('pylast.LastFMNetwork')
    @patch('dotenv.load_dotenv', return_value=True)
    def test_setup_lastfm_client(self, mock_dotenv, mock_lastfm):
        with patch.dict('os.environ', {
            'LASTFM_API_KEY': 'test_key',
            'LASTFM_API_SECRET': 'test_secret'
        }):
            with patch('builtins.print') as mock_print:
                self.recommender.setup_lastfm_client()
                mock_lastfm.assert_called_once_with(
                    api_key='test_key',
                    api_secret='test_secret'
                )
                self.assertIsNotNone(self.recommender.network)
                mock_print.assert_called_once_with("Successfully connected to Last.fm API!")

    @patch('pylast.LastFMNetwork')
    def test_get_artist_features(self, mock_lastfm):
        mock_artist = MagicMock()
        mock_artist.get_top_tags.return_value = [
            MagicMock(item=MagicMock(get_name=lambda: 'rock')),
            MagicMock(item=MagicMock(get_name=lambda: 'pop'))
        ]
        mock_artist.get_similar.return_value = [
            MagicMock(name='Similar Artist 1'),
            MagicMock(name='Similar Artist 2')
        ]
        mock_artist.get_listener_count.return_value = 1000
        
        mock_lastfm.return_value.get_artist.return_value = mock_artist
        self.recommender.network = mock_lastfm.return_value
        
        features = self.recommender.get_artist_features('Test Artist')
        
        self.assertIsInstance(features, dict)
        self.assertIn('tags', features)
        self.assertIn('similar_artists', features)
        self.assertIn('global_listeners', features)
        self.assertEqual(len(features['tags']), 2)
        self.assertEqual(len(features['similar_artists']), 2)
        self.assertEqual(features['global_listeners'], 1000)

    def test_process_listening_history(self):
        self.recommender.load_streaming_history(self.test_data_dir)
        
        with patch.object(self.recommender, 'get_artist_features') as mock_get_features:
            mock_get_features.return_value = {
                'tags': {'rock', 'pop'},
                'similar_artists': {'Similar Artist 1', 'Similar Artist 2'},
                'global_listeners': 1000
            }
            
            self.recommender.process_listening_history()
            
            self.assertIsNotNone(self.recommender.artist_features)
            self.assertGreater(len(self.recommender.artist_features), 0)
            
            # Check if basic statistics were collected
            artist_data = self.recommender.artist_features['Test Artist 1']
            self.assertGreater(artist_data['play_count'], 0)
            self.assertGreater(artist_data['total_ms'], 0)
            self.assertGreater(len(artist_data['unique_tracks']), 0)

    def test_get_recommendations(self):
        # Setup mock data
        self.recommender.artist_features = {
            'Test Artist 1': {
                'play_count': 10,
                'total_ms': 2400000,
                'unique_tracks': {'Track 1'},
                'tags': {'rock', 'pop'},
                'similar_artists': {'Rec Artist 1', 'Rec Artist 2'},
                'global_listeners': 1000
            }
        }
        
        with patch.object(self.recommender, 'get_artist_features') as mock_get_features:
            mock_get_features.return_value = {
                'tags': {'rock', 'pop'},
                'similar_artists': set(),
                'global_listeners': 2000
            }
            
            recommendations = self.recommender.get_recommendations(top_n=2)
            
            self.assertIsInstance(recommendations, list)
            self.assertLessEqual(len(recommendations), 2)
            if recommendations:
                self.assertIn('artist', recommendations[0])
                self.assertIn('score', recommendations[0])

    @patch('dotenv.load_dotenv', return_value=True)
    def test_error_handling(self, mock_dotenv):
        # Test missing data error
        recommender = MusicRecommender()
        with self.assertRaises(ValueError):
            recommender.process_listening_history()
        
        # Test missing API keys error
        recommender = MusicRecommender()
        with patch.dict('os.environ', {}, clear=True):
            with patch('builtins.print') as mock_print:
                with self.assertRaises(ValueError):
                    recommender.setup_lastfm_client()
                mock_print.assert_not_called()

    def test_clustering(self):
        self.recommender.load_streaming_history(self.test_data_dir)
        
        with patch.object(self.recommender, 'get_artist_features') as mock_get_features:
            mock_get_features.return_value = {
                'tags': {'rock', 'pop'},
                'similar_artists': {'Similar Artist 1'},
                'global_listeners': 1000
            }
            
            self.recommender.process_listening_history()
            
            # Check if clustering was performed
            if len(self.recommender.artist_features) >= 5:
                self.assertIsNotNone(self.recommender.artist_clusters)
                if self.recommender.artist_clusters:
                    self.assertIn('centers', self.recommender.artist_clusters)
                    self.assertIn('artists', self.recommender.artist_clusters)

if __name__ == '__main__':
    unittest.main(verbosity=2)