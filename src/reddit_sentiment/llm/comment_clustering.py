"""
Comment clustering for diverse LLM sampling.

Uses TF-IDF + Mini-Batch K-Means to cluster comments by topic,
then selects representative samples from each cluster to ensure
diverse topics are sent to the LLM for analysis.
"""

from __future__ import annotations

import logging
import re
import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common stopwords to exclude from cluster names (financial context)
EXTRA_STOPWORDS = {
    # Generic conversational words
    'just', 'like', 'got', 'get', 'know', 'use', 'used', 'using',
    'going', 'want', 'need', 'needs', 'think', 'said', 'say', 'way', 'thing',
    'make', 'made', 'really', 'good', 'bad', 'great', 'best', 'worst',
    'look', 'looking', 'looks', 'thanks', 'thank', 'help', 'helped', 'new', 'old',
    'time', 'year', 'years', 'month', 'months', 'day', 'days', 'week', 'ago',
    'people', 'person', 'lot', 'lots', 'much', 'many', 'little', 'bit', 'actually',
    'pretty', 'sure', 'right', 'yes', 'yeah', 'definitely', 'probably',
    'maybe', 'guess', 'mean', 'surprised', 'surprised',
    # Action/process words (not distinctive)
    'performed', 'automatically', 'doing', 'done', 'tried', 'trying',
    'called', 'calling', 'asked', 'asking', 'told', 'saying', 'went', 'come',
    'happened', 'happening', 'started', 'ended', 'worked', 'working',
    # Contractions fragments
    'don', 'doesn', 'didn', 'wasn', 'isn', 'aren', 'wouldn', 'couldn', 'shouldn',
    've', 'll', 're', 'didn', 'won',
    # Finance/card terms (too common - not distinctive)
    'card', 'cards', 'account', 'accounts', 'bank', 'banks', 'banking',
    'credit', 'debit', 'capital', 'one', 'capitalone', 'cap',
    'money', 'pay', 'paid', 'paying', 'payment', 'payments',
    'fee', 'fees', 'rate', 'rates', 'apr', 'interest',
    'membership', 'member', 'customer', 'customers', 'service',
    'income', 'loans', 'loan',
    # Generic report/dispute terms
    'report', 'police', 'fraud', 'dispute', 'issue', 'problem',
    # Reddit/AutoMod references (not topic-specific)
    'sidebar', 'resources', 'automod', 'automoderator', 'wiki', 'faq',
    'subreddit', 'sub', 'post', 'comment', 'thread', 'op',
    # Generic advice phrases
    'contact', 'call', 'check', 'try', 'recommend', 'suggest',
}

# Pattern for mostly-numeric terms (should be excluded from cluster names)
# Matches: "000 miles", "70 000", "50000", "$500", etc.
NUMERIC_PATTERN = re.compile(r'^[\d\s,\.\$%]+$|^\d+\s*\w*\s*\d+$|000|^\d+$')


class CommentClusterer:
    """Cluster comments by topic for diverse sampling."""

    def __init__(
        self,
        min_comments_for_clustering: int = 10,
        max_clusters: int = 5,
        min_clusters: int = 2,
        proximity_weight: float = 0.7,
    ):
        """
        Initialize the clusterer.

        Args:
            min_comments_for_clustering: Minimum comments needed to cluster
            max_clusters: Maximum number of clusters per sentiment group
            min_clusters: Minimum number of clusters
            proximity_weight: Weight for centroid proximity vs engagement (0-1)
        """
        self.min_comments_for_clustering = min_comments_for_clustering
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.proximity_weight = proximity_weight

        if SKLEARN_AVAILABLE:
            # TF-IDF settings optimized for short social media text
            self.vectorizer = TfidfVectorizer(
                max_features=500,        # Limit vocabulary for speed
                min_df=2,                # Ignore rare terms
                max_df=0.95,             # Ignore very common terms
                ngram_range=(1, 2),      # Unigrams and bigrams
                stop_words='english',
            )
        else:
            self.vectorizer = None

    def get_diverse_samples(
        self,
        df: pd.DataFrame,
        n_samples: int = 15,
        text_col: str = 'body',
    ) -> pd.DataFrame:
        """
        Get diverse samples by clustering and selecting representatives.

        Args:
            df: DataFrame with comments
            n_samples: Target number of samples
            text_col: Column containing text

        Returns:
            DataFrame with diverse samples
        """
        if not SKLEARN_AVAILABLE:
            logger.info("Clustering: sklearn not available, using fallback")
            return self._fallback_selection(df, n_samples)

        if len(df) < self.min_comments_for_clustering:
            logger.info(f"Clustering: only {len(df)} comments (< {self.min_comments_for_clustering}), using fallback")
            return self._fallback_selection(df, n_samples)

        # Vectorize
        texts = df[text_col].fillna('').astype(str).tolist()

        # Filter out empty texts
        valid_mask = [len(t.strip()) > 0 for t in texts]
        if sum(valid_mask) < self.min_comments_for_clustering:
            return self._fallback_selection(df, n_samples)

        try:
            X = self.vectorizer.fit_transform(texts)
        except ValueError:
            # All documents are empty or have only stop words
            return self._fallback_selection(df, n_samples)

        # Check if we have enough features
        if X.shape[1] == 0:
            return self._fallback_selection(df, n_samples)

        # Find optimal cluster count
        n_clusters = self._find_optimal_clusters(X)

        # Cluster
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=min(256, len(df)),
        )
        labels = kmeans.fit_predict(X)

        # Select per cluster (distribute samples across clusters)
        per_cluster = max(1, n_samples // n_clusters)

        # Log clustering results
        cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
        logger.info(f"Clustering: {len(df)} comments -> {n_clusters} clusters (sizes: {cluster_sizes}), selecting {per_cluster} per cluster")
        print(f"[CLUSTERING] {len(df)} comments -> {n_clusters} clusters (sizes: {cluster_sizes})", flush=True)

        # Also try streamlit if available
        try:
            import streamlit as st
            st.toast(f"Clustering: {len(df)} → {n_clusters} clusters")
        except:
            pass

        # Get cluster names from top TF-IDF terms
        cluster_names = self._get_cluster_names(labels, X, n_clusters)
        logger.info(f"Clustering: cluster names: {cluster_names}")
        print(f"[CLUSTERING] Topics: {cluster_names}", flush=True)

        return self._select_representatives(
            df, labels, kmeans.cluster_centers_, X, per_cluster, cluster_names
        )

    def _fallback_selection(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Fallback selection when clustering isn't possible."""
        if len(df) == 0:
            return df

        n_select = min(n_samples, len(df))

        if 'score' in df.columns:
            result = df.nlargest(n_select, 'score').copy()
        else:
            result = df.head(n_select).copy()

        # Add placeholder cluster columns for consistency
        result['cluster_name'] = 'Unclustered'
        result['cluster_id'] = 0
        return result

    def _find_optimal_clusters(self, X) -> int:
        """Find optimal cluster count using silhouette score."""
        n_samples = X.shape[0]
        max_k = min(self.max_clusters, n_samples // 3)
        min_k = self.min_clusters

        if max_k < min_k:
            return max(1, min(min_k, n_samples // 2))

        best_k, best_score = min_k, -1

        for k in range(min_k, max_k + 1):
            try:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=3,
                    batch_size=min(256, n_samples),
                )
                labels = kmeans.fit_predict(X)

                # Skip if degenerate clustering
                unique_labels = len(set(labels))
                if unique_labels < k:
                    continue

                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score
            except ValueError:
                continue

        return best_k

    def _get_cluster_names(
        self,
        labels: np.ndarray,
        X,
        n_clusters: int,
        top_n_terms: int = 2,
    ) -> dict[int, str]:
        """Extract descriptive names for each cluster using top TF-IDF terms.

        Prefers bigrams (two-word phrases) over single words for better readability.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_names = {}

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if not cluster_mask.any():
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
                continue

            # Sum TF-IDF scores for this cluster
            cluster_tfidf = X[cluster_mask].sum(axis=0).A1  # Convert to 1D array

            # Get top terms, preferring bigrams
            sorted_indices = cluster_tfidf.argsort()[::-1]
            bigrams = []
            unigrams = []

            for idx in sorted_indices:
                term = feature_names[idx]
                term_lower = term.lower()

                # Skip stopwords and short terms
                if len(term) < 3:
                    continue

                # Skip numeric-heavy terms (e.g., "000 miles", "70 000")
                if NUMERIC_PATTERN.search(term):
                    continue

                # Check if ALL words in term are stopwords
                words = term_lower.split()
                if all(w in EXTRA_STOPWORDS for w in words):
                    continue

                # Skip terms with numbers at the start (e.g., "50000 points")
                if words and words[0].isdigit():
                    continue

                # Prefer bigrams (two-word phrases)
                if ' ' in term and len(bigrams) < top_n_terms:
                    bigrams.append(term)
                elif ' ' not in term and len(unigrams) < top_n_terms:
                    unigrams.append(term)

                # Stop if we have enough
                if len(bigrams) >= top_n_terms:
                    break

            # Use bigrams if available, otherwise unigrams
            top_terms = bigrams if bigrams else unigrams[:top_n_terms]

            if top_terms:
                # Join with " & " for readability
                cluster_names[cluster_id] = " & ".join(top_terms)
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id}"

        return cluster_names

    def _select_representatives(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        centroids: np.ndarray,
        X,
        per_cluster: int,
        cluster_names: Optional[dict[int, str]] = None,
    ) -> pd.DataFrame:
        """Select representative comments using hybrid scoring."""
        selected_parts = []

        for cluster_id in range(len(centroids)):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_df = df.iloc[cluster_indices].copy()

            # Add cluster name to selected comments
            if cluster_names:
                cluster_df['cluster_name'] = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
                cluster_df['cluster_id'] = cluster_id
            cluster_X = X[cluster_mask]

            # Distance to centroid
            centroid = centroids[cluster_id].reshape(1, -1)
            distances = cosine_distances(cluster_X, centroid).flatten()

            # Normalize proximity (invert distance - lower distance = higher score)
            if distances.max() > distances.min():
                proximity = 1 - (distances - distances.min()) / (distances.max() - distances.min())
            else:
                proximity = np.ones(len(distances))

            # Engagement score (Reddit upvotes)
            if 'score' in cluster_df.columns:
                scores = cluster_df['score'].values.astype(float)
                if scores.max() > scores.min():
                    engagement = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    engagement = np.ones(len(scores))
            else:
                engagement = np.zeros(len(cluster_df))

            # Hybrid score: proximity + engagement
            hybrid = self.proximity_weight * proximity + (1 - self.proximity_weight) * engagement

            # Select top comments from this cluster
            n_select = min(per_cluster, len(cluster_df))
            top_idx = np.argsort(hybrid)[-n_select:]
            selected_parts.append(cluster_df.iloc[top_idx])

        if not selected_parts:
            return df.head(0)  # Empty DataFrame with same columns

        return pd.concat(selected_parts, ignore_index=True)


def is_clustering_available() -> bool:
    """Check if clustering dependencies are available."""
    return SKLEARN_AVAILABLE
