import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
from typing import Dict, List

class ClusteringAnalyzer:
    def perform_clustering(self, artist_features: Dict, clustering_type: str = "behavior") -> Dict:
        """Performs clustering analysis on given features"""
        features, artists = self.prepare_features(artist_features, clustering_type)
        return self._perform_single_clustering(features, artists, artist_features, clustering_type + "_based")

    def _perform_single_clustering(self, features, artists, artist_features: Dict, clustering_type: str) -> Dict:
        """Performs clustering for a single type of features"""
        if not features:
            return {}
        
        features_array = np.array(features)
        features_scaled = StandardScaler().fit_transform(features_array)
        
        # Dynamiczne określanie liczby klastrów
        n_samples = len(features)
        min_clusters = 3  # minimum 3 klastry
        max_clusters = min(8, n_samples // 50)  # maksymalnie 8 klastrów lub 1 klaster na 50 próbek
        max_clusters = max(min_clusters, max_clusters)  # nie mniej niż min_clusters
        
        best_n_clusters = min_clusters
        best_score = float('-inf')
        best_kmeans = None
        clustering_metrics = []
        
        # Próbujemy różne liczby klastrów
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Oblicz metryki klastrów
            cluster_sizes = np.bincount(clusters)
            min_size = np.min(cluster_sizes)
            max_size = np.max(cluster_sizes)
            size_ratio = min_size / max_size
            
            sil_score = silhouette_score(features_scaled, clusters)
            cal_score = calinski_harabasz_score(features_scaled, clusters)
            dav_score = davies_bouldin_score(features_scaled, clusters)
            
            # Zmodyfikowana metryka oceny z większym naciskiem na zbalansowanie
            combined_score = (
                0.3 * sil_score +  # jakość klastrów
                0.2 * (1 / (1 + dav_score)) +  # separacja klastrów
                0.5 * size_ratio  # zbalansowanie klastrów (zwiększona waga)
            )
            
            clustering_metrics.append({
                'n_clusters': n_clusters,
                'silhouette': sil_score,
                'calinski_harabasz': cal_score,
                'davies_bouldin': dav_score,
                'min_cluster_size': min_size,
                'size_ratio': size_ratio,
                'combined_score': combined_score
            })
            
            if combined_score > best_score:
                best_score = combined_score
                best_n_clusters = n_clusters
                best_kmeans = kmeans
        
        # Użyj najlepszego modelu
        clusters = best_kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_data = defaultdict(lambda: {
            'artists': [],
            'genres': defaultdict(int),
            'avg_popularity': 0,
            'avg_plays': 0,
            'size': 0,
            'clustering_type': clustering_type,
            'metrics': clustering_metrics[best_n_clusters - 3],
            'clustering_history': clustering_metrics
        })
        
        for artist, cluster in zip(artists, clusters):
            data = artist_features[artist]
            cluster_info = cluster_data[int(cluster)]
            cluster_info['artists'].append(artist)
            cluster_info['avg_popularity'] += data['global_popularity']
            cluster_info['avg_plays'] += data['play_count']
            cluster_info['size'] += 1
            for genre in data['genres']:
                cluster_info['genres'][genre] += 1
        
        # Normalize cluster statistics
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

    def prepare_features(self, artist_features: Dict, clustering_type: str = "behavior") -> tuple:
        """Prepares feature vectors and artist lists for clustering"""
        features = []
        artists = []
        genre_features = []
        all_genres = set()
        
        # Collect all unique genres first
        for data in artist_features.values():
            all_genres.update(data['genres'])
        genre_list = sorted(list(all_genres))
        
        # Prepare feature vectors
        min_plays = 5  # zmniejszamy próg odtworzeń
        for artist, data in artist_features.items():
            if data['play_count'] >= min_plays:
                if clustering_type == "behavior":
                    # Dodajemy więcej cech dla lepszego rozróżnienia
                    feature_vector = [
                        np.log1p(data['play_count']),
                        np.log1p(data['total_ms'] / (1000 * 60 * 60)),
                        np.log1p(len(data['unique_tracks'])),
                        data['global_popularity'] / 100,
                        len(data['genres']) / max(1, len(all_genres)),
                        np.mean(data['listening_gaps']) if data['listening_gaps'] else 0,
                        len(data['unique_tracks']) / max(1, data['play_count']),  # różnorodność utworów
                        data['total_ms'] / max(1, data['play_count']),  # średni czas utworu
                    ]
                    features.append(feature_vector)
                    artists.append(artist)
                else:  # genre-based
                    # Normalizujemy wektor gatunków
                    weight = np.log1p(data['play_count'])
                    genre_vector = [weight if genre in data['genres'] else 0 for genre in genre_list]
                    if sum(genre_vector) > 0:
                        genre_vector = np.array(genre_vector) / max(1, sum(genre_vector))  # normalizacja
                        genre_features.append(genre_vector)
                        artists.append(artist)
        
        if clustering_type == "behavior":
            return features, artists
        else:
            return genre_features, artists 