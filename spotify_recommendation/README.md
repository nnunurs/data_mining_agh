# Music Recommendation System

A sophisticated music recommendation system that analyzes your Spotify listening history and uses Last.fm data to generate personalized artist recommendations.

## Features

- Analyzes Spotify listening history from JSON files
- Integrates with Last.fm API for artist metadata
- Uses machine learning (K-means clustering) for artist categorization
- Provides personalized recommendations based on:
  - Listening patterns
  - Genre preferences
  - Artist similarity
  - Global popularity

## Requirements

- Python 3.8+
- Last.fm API credentials

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Last.fm API credentials:
```
LASTFM_API_KEY=your_api_key
LASTFM_API_SECRET=your_api_secret
```

## Usage

1. Place your Spotify listening history JSON files in the `data/` directory. Files have to be JSON files downloaded from Spotify, their names should look like `Streaming_History_Audio_{time period}.json`
2. Run the recommender:
```bash
python spotify_recommender.py
```

## How it works

1. **Data Loading**
   - Loads and combines Spotify listening history from JSON files
   - Filters out non-music content (podcasts, etc.)

2. **Data Processing**
   - Analyzes listening patterns (play counts, time spent, etc.)
   - Retrieves artist metadata from Last.fm (tags, similar artists)
   - Normalizes and processes features for clustering

3. **Artist Clustering**
   - Uses K-means clustering to group artists based on:
     - Musical characteristics (tags)
     - Listening statistics
     - Popularity metrics

4. **Recommendation Generation**
   - Identifies potential new artists through Last.fm similarities
   - Scores candidates based on:
     - Tag similarity with favorite genres
     - Artist popularity
     - Cluster alignment

## Output

The system provides:
- Analysis of your listening preferences
- Preferred listening hours
- Favorite music genres
- Detailed artist recommendations with:
  - Genre tags
  - Popularity metrics
  - Similarity scores

## Note

This system requires Spotify's "Extended Streaming History" which can be requested through your Spotify account settings. 