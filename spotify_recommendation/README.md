# Spotify Music Analysis & Recommendation

A tool for analyzing Spotify listening history and generating personalized music recommendations.

## Features

### Listening Analysis
- Time pattern analysis (daily/weekly listening distribution)
- Listening session lengths
- Genre trends
- New genre discovery tracking
- Artist clustering based on listener behavior

### Recommendations
- Popular artist recommendations
- Hidden gems discovery (less known artists)
- Genre similarity-based recommendations

### Visualizations
The program generates the following visualizations in the `plots/` folder:
- `daily_distribution.png` - Daily listening distribution
- `weekly_distribution.png` - Weekly listening distribution
- `session_lengths.png` - Listening session length distribution
- `top_genres.png` - Top 10 most listened genres
- `genre_flows.png` - Most common genre transitions
- `genre_exploration.png` - Genre discovery trends
- `clusters.png` - Artist clusters visualization (PCA)

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, spotipy
- Spotify account and listening history

## Installation and Setup
1. Clone the repository
2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn spotipy python-dotenv
```
3. Create a `.env` file in the root directory with your Spotify API credentials:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Usage
```bash
python spotify_analyzer.py
```

## Project Structure
```
spotify_recommendation/
├── data/                     # Input data and cache
│   ├── cache/               # Artist data cache
│   └── plots/               # Generated visualizations
├── spotify_analyzer.py       # Main script
├── clustering.py            # Clustering analysis
├── listening_stats.py       # Listening statistics analysis
├── visualization.py         # Visualization generation
├── spotify_client.py        # Spotify API client
└── README.md
```

## Technical Details
- Behavioral clustering uses the following features:
  - Play count
  - Total listening time
  - Unique tracks count
  - Global popularity
  - Genre diversity
  - Average gap between listens
  - Track diversity
  - Average track duration

- Visualizations use matplotlib and seaborn libraries
- Data is automatically cached for faster subsequent analyses

## Notes
- System requires Spotify listening history in JSON format
- History files should be placed in the `data/` directory
- Cache is stored in `data/cache/` 