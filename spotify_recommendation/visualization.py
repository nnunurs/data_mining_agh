import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from collections import defaultdict

class Visualizer:
    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir
    
    def generate_all_visualizations(self, analysis: Dict) -> None:
        """Generates and saves all visualizations"""
        self.visualize_listening_stats(analysis)

    def visualize_listening_stats(self, analysis: Dict) -> None:
        """Visualizes listening statistics"""
        if 'temporal' in analysis:
            temp = analysis['temporal']
            
            # Rozkład dzienny
            plt.figure(figsize=(12, 6))
            hours = list(range(24))
            counts = [temp['daily'].get(hour, 0) for hour in hours]
            total = sum(counts)
            percentages = [count/total*100 for count in counts]
            
            plt.plot(hours, percentages, 'o-', linewidth=2)
            plt.fill_between(hours, percentages, alpha=0.3)
            plt.title('Daily Listening Distribution')
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage of Listening Time')
            plt.grid(True)
            plt.xticks(hours)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'daily_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Rozkład tygodniowy
            plt.figure(figsize=(10, 6))
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            counts = [temp['weekday'].get(day, 0) for day in days]
            total = sum(counts)
            percentages = [count/total*100 for count in counts]
            
            plt.bar(days, percentages)
            plt.title('Weekly Listening Distribution')
            plt.xlabel('Day of Week')
            plt.ylabel('Percentage of Listening Time')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'weekly_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Długości sesji
            if 'session_length' in temp and temp['session_length']:
                plt.figure(figsize=(10, 6))
                plt.hist(temp['session_length'], bins=50, edgecolor='black')
                plt.title('Distribution of Listening Session Lengths')
                plt.xlabel('Session Length (minutes)')
                plt.ylabel('Number of Sessions')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'session_lengths.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Trendy gatunków
        if 'genres' in analysis:
            genres = analysis['genres']
            
            # Top gatunki
            plt.figure(figsize=(12, 6))
            top_genres = sorted(genres['genre_expertise'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
            genre_names, values = zip(*top_genres)
            total = sum(values)
            percentages = [value/total*100 for value in values]
            
            plt.bar(genre_names, percentages)
            plt.title('Top 10 Genres by Listening Time')
            plt.xlabel('Genre')
            plt.ylabel('Percentage of Total Listening Time')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'top_genres.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Przepływ gatunków jako macierz
            genre_flows = defaultdict(lambda: defaultdict(int))
            for from_genre, transitions in genres['genre_flow'].items():
                for to_genre, count in transitions.items():
                    genre_flows[from_genre][to_genre] = count
            
            # Znajdź top N najczęściej występujących gatunków
            top_n = 15
            all_genres = set()
            for from_genre, transitions in genre_flows.items():
                all_genres.add(from_genre)
                all_genres.update(transitions.keys())
            
            genre_counts = defaultdict(int)
            for genre in all_genres:
                for transitions in genre_flows.values():
                    genre_counts[genre] += sum(count for g, count in transitions.items() if g == genre)
            
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_genre_names = [g[0] for g in top_genres]
            
            # Utwórz macierz przejść
            matrix = np.zeros((top_n, top_n))
            for i, from_genre in enumerate(top_genre_names):
                for j, to_genre in enumerate(top_genre_names):
                    matrix[i, j] = genre_flows[from_genre][to_genre]
            
            # Normalizuj macierz (opcjonalnie)
            matrix = matrix / matrix.sum() * 100  # jako procent wszystkich przejść
            
            # Wizualizacja
            plt.figure(figsize=(15, 12))
            mask = matrix == 0
            
            # Użyj kolorowej mapy z białym tłem dla zerowych wartości
            cmap = sns.color_palette("rocket_r", as_cmap=True)
            sns.heatmap(matrix, 
                        xticklabels=top_genre_names,
                        yticklabels=top_genre_names,
                        cmap=cmap,
                        mask=mask,
                        annot=True,  # pokaż wartości
                        fmt='.1f',   # format liczb
                        square=True, # kwadratowe komórki
                        cbar_kws={'label': 'Transition Percentage (%)'},
                        linewidths=0.5)
            
            plt.title('Genre Transition Matrix')
            plt.xlabel('To Genre')
            plt.ylabel('From Genre')
            
            # Obróć etykiety dla lepszej czytelności
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'genre_flows.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Trendy odkrywania
        if 'discovery' in analysis and 'exploration_patterns' in analysis['discovery']:
            patterns = analysis['discovery']['exploration_patterns']
            if patterns:
                # Konwertuj stringi dat na obiekty datetime
                months = pd.to_datetime([p['month'] for p in patterns], format='%Y-%m')
                new_genres = [len(p['new_genres']) for p in patterns]
                rates = [p['exploration_rate']*100 for p in patterns]
                
                # Jeden wykres z dwiema osiami Y
                fig, ax1 = plt.subplots(figsize=(15, 8))
                
                # Pierwsza oś Y dla liczby nowych gatunków
                color1 = '#FF69B4'  # różowy
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Number of New Genres', color=color1)
                line1 = ax1.plot(months, new_genres, 'o-', color=color1, label='New Genres')
                ax1.tick_params(axis='y', labelcolor=color1)
                
                # Druga oś Y dla wskaźnika eksploracji
                ax2 = ax1.twinx()
                color2 = '#4B0082'  # indygo
                ax2.set_ylabel('Exploration Rate (%)', color=color2)
                line2 = ax2.plot(months, rates, 'o-', color=color2, label='Exploration Rate')
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Formatowanie osi X
                plt.gcf().autofmt_xdate()  # Automatyczne formatowanie dat
                ax1.xaxis.set_major_locator(YearLocator())
                ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
                ax1.xaxis.set_minor_locator(MonthLocator())
                
                # Tytuł i siatka
                plt.title('Genre Discovery and Exploration Over Time')
                ax1.grid(True, alpha=0.3)
                
                # Legenda
                lines1 = line1 + line2
                labels = [l.get_label() for l in lines1]
                ax1.legend(lines1, labels, loc='upper right')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'genre_exploration.png', dpi=300, bbox_inches='tight')
                plt.close()

class DataVisualizer:
    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [10, 6]
    
    def visualize_clusters(self, clusters: Dict, features: List, artists: List, title: str = "Cluster Visualization"):
        """Visualizes clusters using PCA for dimensionality reduction"""
        if not features or not artists:
            return
        
        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'artist': artists,
            'cluster': [0] * len(artists)
        })
        
        # Assign cluster numbers
        for cluster_id, cluster_info in clusters.items():
            for artist in cluster_info['artists']:
                df.loc[df['artist'] == artist, 'cluster'] = cluster_id
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot points
        scatter = plt.scatter(df['x'], df['y'], c=df['cluster'], 
                            cmap='viridis', alpha=0.6, s=100)
        
        # Add cluster centers and annotations
        for cluster_id in clusters:
            cluster_points = df[df['cluster'] == cluster_id]
            center_x = cluster_points['x'].mean()
            center_y = cluster_points['y'].mean()
            plt.scatter(center_x, center_y, c='red', marker='x', s=200, 
                       linewidths=3, label=f'Cluster {cluster_id} Center')
            
            # Add annotation with cluster info
            cluster_info = clusters[cluster_id]
            info_text = f"Cluster {cluster_id}\n"
            info_text += f"Size: {cluster_info['size']}\n"
            info_text += f"Avg plays: {cluster_info['avg_plays']:.1f}\n"
            info_text += f"Top genre: {cluster_info['top_genres'][0][0]}"
            
            plt.annotate(info_text, (center_x, center_y), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Add labels and title
        plt.title(f'{title}\nPCA visualization of artist clusters')
        plt.xlabel(f'First Principal Component\nExplained variance: {pca.explained_variance_ratio_[0]:.2%}')
        plt.ylabel(f'Second Principal Component\nExplained variance: {pca.explained_variance_ratio_[1]:.2%}')
        
        # Add colorbar
        plt.colorbar(scatter, label='Cluster')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.plots_dir / 'clusters.png', dpi=300, bbox_inches='tight')
        plt.close() 