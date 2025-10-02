import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# TMDb API Configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        max-width: 100%;
    }
    .movie-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .movie-title {
        color: #e50914;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .movie-info {
        color: #b3b3b3;
        font-size: 14px;
        margin: 5px 0;
    }
    .rating {
        color: #ffd700;
        font-size: 18px;
        font-weight: bold;
    }
    .pagination-bottom {
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript to scroll to top on page load
st.markdown("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
""", unsafe_allow_html=True)

class MovieRecommender:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache_file = 'movie_cache.pkl'
        self.movie_data = self.load_cache()
        self.session = self._create_session()
        self.search_history_file = 'search_history.pkl'
        self.user_search_history = self.load_search_history()
        self.liked_movies_file = 'liked_movies.pkl'
        self.liked_movies = self.load_liked_movies()
        
    def _create_session(self):
        """Create a requests session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
        
    def load_cache(self):
        """Load cached movie data if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        """Save movie data to cache"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.movie_data, f)
    
    def load_search_history(self):
        """Load user search history if available"""
        if os.path.exists(self.search_history_file):
            try:
                with open(self.search_history_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return []
        return []
    
    def save_search_history(self):
        """Save user search history"""
        try:
            with open(self.search_history_file, 'wb') as f:
                pickle.dump(self.user_search_history, f)
        except Exception as e:
            pass
    
    def load_liked_movies(self):
        """Load user's liked movies if available"""
        if os.path.exists(self.liked_movies_file):
            try:
                with open(self.liked_movies_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_liked_movies(self):
        """Save user's liked movies"""
        try:
            with open(self.liked_movies_file, 'wb') as f:
                pickle.dump(self.liked_movies, f)
        except Exception as e:
            pass
    
    def like_movie(self, movie_id, movie_data, mode=None):
        """Add a movie to liked list"""
        self.liked_movies[movie_id] = {
            'movie_data': movie_data,
            'timestamp': datetime.now(),
            'mode': mode or 'Unknown'
        }
        self.save_liked_movies()
    
    def unlike_movie(self, movie_id):
        """Remove a movie from liked list"""
        if movie_id in self.liked_movies:
            del self.liked_movies[movie_id]
            self.save_liked_movies()
    
    def is_movie_liked(self, movie_id):
        """Check if a movie is liked"""
        return movie_id in self.liked_movies
    
    def add_to_search_history(self, query, movie_ids=None, languages=None):
        """Track user search queries, viewed movies, and languages"""
        search_entry = {
            'query': query,
            'movie_ids': movie_ids or [],
            'languages': languages or [],
            'timestamp': datetime.now()
        }
        self.user_search_history.append(search_entry)
        # Keep only last 50 searches
        if len(self.user_search_history) > 50:
            self.user_search_history = self.user_search_history[-50:]
        self.save_search_history()
    
    def fetch_trending_movies(self, page=1, language_code=None, year=None):
        """Fetch trending movies from TMDb with optional filters"""
        if language_code or year:
            # Use discover endpoint for filtering
            url = f"{TMDB_BASE_URL}/discover/movie"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'accept': 'application/json'
            }
            params = {
                'sort_by': 'popularity.desc',
                'page': page
            }
            if language_code:
                params['with_original_language'] = language_code
            if year:
                params['primary_release_year'] = year
        else:
            # Use trending endpoint when no filters
            url = f"{TMDB_BASE_URL}/trending/movie/week"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'accept': 'application/json'
            }
            params = {'page': page}
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching trending movies: {e}")
            return []
    
    def fetch_popular_movies(self, page=1):
        """Fetch popular movies from TMDb"""
        url = f"{TMDB_BASE_URL}/movie/popular"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {'page': page}
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching popular movies: {e}")
            return []
    
    def search_movies(self, query, page=1):
        """Search for movies by title"""
        url = f"{TMDB_BASE_URL}/search/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {
            'query': query,
            'page': page
        }
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['results'], data.get('total_pages', 1)
        except Exception as e:
            st.error(f"Error searching movies: {e}")
            return [], 1
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a movie"""
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {'append_to_response': 'credits,keywords,similar'}
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching movie details: {e}")
            return None
    
    def get_movie_by_genre(self, genre_id, page=1):
        """Get movies by genre"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching movies by genre: {e}")
            return []
    
    def get_genres(self):
        """Get list of all genres"""
        url = f"{TMDB_BASE_URL}/genre/movie/list"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()['genres']
        except Exception as e:
            st.error(f"Error fetching genres: {e}")
            return []
    
    def get_movie_by_language(self, language_code, page=1):
        """Get movies by language"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {
            'with_original_language': language_code,
            'sort_by': 'popularity.desc',
            'page': page
        }
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching movies by language: {e}")
            return []
    
    def get_movie_by_year(self, year, page=1):
        """Get movies by release year"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {
            'primary_release_year': year,
            'sort_by': 'popularity.desc',
            'page': page
        }
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching movies by year: {e}")
            return []
    
    def get_movies_with_filters(self, genre_id=None, language_code=None, year=None, page=1):
        """Get movies with multiple filters combined"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        params = {
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        # Add filters if provided
        if genre_id:
            params['with_genres'] = genre_id
        if language_code:
            params['with_original_language'] = language_code
        if year:
            params['primary_release_year'] = year
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching movies with filters: {e}")
            return []
    
    def get_latest_movies(self, page=1, language_codes=None, include_dubbed=True):
        """Get all movies sorted by release date (newest first)"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        # Get all movies sorted by release date descending (newest to oldest)
        params = {
            'sort_by': 'primary_release_date.desc',
            'page': page
        }
        
        # Add language filter if provided (supports multiple languages)
        # Use 'with_original_language' for original + 'with_release_type' to include dubbed versions
        if language_codes:
            if include_dubbed:
                # This will include movies available in the selected languages (original + dubbed)
                params['with_original_language'] = '|'.join(language_codes)
            else:
                params['with_original_language'] = '|'.join(language_codes)
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching latest movies: {e}")
            return []
    
    def get_dubbed_movies(self, target_language, exclude_original=True, page=1):
        """Get popular movies that may be available dubbed in target language"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json'
        }
        
        # Get popular movies, optionally excluding those originally in target language
        params = {
            'sort_by': 'popularity.desc',
            'page': page,
            'vote_count.gte': 100  # Only movies with significant votes (more likely to have dubs)
        }
        
        if exclude_original:
            # Exclude movies originally in the target language
            params['with_original_language'] = f'!{target_language}'
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            st.error(f"Error fetching dubbed movies: {e}")
            return []
    
    def process_poster_with_opencv(self, image_url):
        """Process movie poster using OpenCV for visual analysis"""
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Extract dominant colors
            pixels = img_bgr.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # K-means clustering to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 5
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Get dominant color
            dominant_color = centers[0].astype(int)
            
            # Calculate brightness
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Edge detection for complexity
            edges = cv2.Canny(gray, 100, 200)
            complexity = np.count_nonzero(edges) / edges.size
            
            return {
                'dominant_color': dominant_color.tolist(),
                'brightness': float(brightness),
                'complexity': float(complexity)
            }
        except Exception as e:
            return None
    
    def get_content_based_recommendations(self, movie_id, movies_list, top_n=10):
        """Get recommendations based on content similarity"""
        target_movie = self.get_movie_details(movie_id)
        if not target_movie:
            return []
        
        # Create feature string for target movie
        target_features = self.create_feature_string(target_movie)
        
        # Create feature strings for all movies
        movie_features = []
        valid_movies = []
        
        for movie in movies_list:
            if movie['id'] != movie_id:
                details = self.get_movie_details(movie['id'])
                if details:
                    feature_str = self.create_feature_string(details)
                    movie_features.append(feature_str)
                    valid_movies.append(details)
        
        if not movie_features:
            return []
        
        # Calculate similarity using TF-IDF
        all_features = [target_features] + movie_features
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_features)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get top N similar movies
        similar_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = [valid_movies[i] for i in similar_indices]
        
        return recommendations
    
    def create_feature_string(self, movie):
        """Create a feature string from movie details for similarity calculation"""
        features = []
        
        # Add genres
        if 'genres' in movie:
            features.extend([g['name'] for g in movie['genres']])
        
        # Add keywords
        if 'keywords' in movie and 'keywords' in movie['keywords']:
            features.extend([k['name'] for k in movie['keywords']['keywords'][:10]])
        
        # Add director
        if 'credits' in movie and 'crew' in movie['credits']:
            directors = [c['name'] for c in movie['credits']['crew'] if c['job'] == 'Director']
            features.extend(directors)
        
        # Add top cast
        if 'credits' in movie and 'cast' in movie['credits']:
            features.extend([c['name'] for c in movie['credits']['cast'][:5]])
        
        # Add overview words
        if 'overview' in movie:
            features.append(movie['overview'])
        
        return ' '.join(features)
    
    def get_recommendations_from_history(self, top_n=20):
        """Get personalized recommendations based on user search history and liked movies"""
        if not self.user_search_history and not self.liked_movies:
            # No history, return trending movies
            return self.fetch_trending_movies(page=1)
        
        # Collect all movie IDs and languages from search history
        all_movie_ids = []
        all_languages = []
        for entry in self.user_search_history:
            all_movie_ids.extend(entry.get('movie_ids', []))
            all_languages.extend(entry.get('languages', []))
        
        # Count language frequency
        language_frequency = {}
        for lang in all_languages:
            if lang:
                language_frequency[lang] = language_frequency.get(lang, 0) + 1
        
        # Get most frequently searched languages (threshold: 2+ times)
        preferred_languages = [lang for lang, count in language_frequency.items() if count >= 2]
        # Sort by frequency
        preferred_languages.sort(key=lambda x: language_frequency[x], reverse=True)
        
        # Get details of recently viewed movies
        recent_movies = []
        for movie_id in list(set(all_movie_ids))[-10:]:  # Last 10 unique movies
            details = self.get_movie_details(movie_id)
            if details:
                recent_movies.append(details)
        
        # Add liked movies to analysis (they have higher priority)
        liked_movie_details = []
        for movie_id, data in self.liked_movies.items():
            movie_data = data.get('movie_data')
            if movie_data:
                liked_movie_details.append(movie_data)
                # Also get full details for better recommendations
                details = self.get_movie_details(movie_id)
                if details:
                    recent_movies.append(details)
        
        if not recent_movies and not preferred_languages:
            # No movie details or language preference, return trending
            return self.fetch_trending_movies(page=1)
        
        # Extract genres, keywords, and languages from user's history and liked movies
        user_genres = set()
        user_keywords = set()
        
        # Prioritize genres from liked movies
        for movie in liked_movie_details:
            genre_ids = movie.get('genre_ids', [])
            user_genres.update(genre_ids)
            # Add language from liked movies
            lang = movie.get('original_language')
            if lang and lang not in preferred_languages:
                preferred_languages.append(lang)
        
        for movie in recent_movies:
            if 'genres' in movie:
                for genre in movie['genres']:
                    user_genres.add(genre['id'])
            if 'keywords' in movie and 'keywords' in movie['keywords']:
                for keyword in movie['keywords']['keywords'][:5]:
                    user_keywords.add(keyword['name'])
        
        # Get movies based on user's preferred languages
        recommended_movies = []
        if preferred_languages:
            # Get movies from top 2 preferred languages
            for lang_code in preferred_languages[:2]:
                lang_movies = self.get_movie_by_language(lang_code, page=1)
                recommended_movies.extend(lang_movies[:15])
        
        # Get movies based on user's preferred genres
        if user_genres:
            # Get movies from top 2 genres
            for genre_id in list(user_genres)[:2]:
                genre_movies = self.get_movie_by_genre(genre_id, page=1)
                recommended_movies.extend(genre_movies[:10])
        
        # Add trending movies (with language filter if available)
        if preferred_languages:
            trending = self.fetch_trending_movies(page=1, language_code=preferred_languages[0])
        else:
            trending = self.fetch_trending_movies(page=1)
        recommended_movies.extend(trending[:10])
        
        # Remove duplicates and movies already in history
        seen_ids = set(all_movie_ids)
        unique_recommendations = []
        for movie in recommended_movies:
            if movie['id'] not in seen_ids:
                unique_recommendations.append(movie)
                seen_ids.add(movie['id'])
        
        # Score and sort recommendations
        scored_movies = []
        liked_genres = set()
        for liked_movie in liked_movie_details:
            liked_genres.update(liked_movie.get('genre_ids', []))
        
        for movie in unique_recommendations:
            score = 0
            
            # BONUS: Extra points if genres match liked movies (highest priority)
            if liked_genres:
                movie_genre_ids = movie.get('genre_ids', [])
                liked_genre_matches = len(set(movie_genre_ids) & liked_genres)
                score += liked_genre_matches * 8  # 8 points per liked genre match
            
            # Score based on language match
            movie_language = movie.get('original_language', '')
            if movie_language in preferred_languages:
                # Higher score for more frequently searched languages
                lang_rank = preferred_languages.index(movie_language)
                score += (10 - lang_rank * 2)  # 10 points for top language, 8 for second, etc.
            
            # Score based on genre match
            movie_genre_ids = movie.get('genre_ids', [])
            genre_matches = len(set(movie_genre_ids) & user_genres)
            score += genre_matches * 5
            
            # Score based on popularity and rating
            score += movie.get('popularity', 0) * 0.02
            score += movie.get('vote_average', 0) * 1.0
            
            # Bonus for recent releases
            release_date = movie.get('release_date', '')
            if release_date:
                try:
                    release_year = int(release_date[:4])
                    current_year = datetime.now().year
                    if current_year - release_year <= 2:
                        score += 3
                except:
                    pass
            
            scored_movies.append((score, movie))
        
        # Sort by score and return top N
        scored_movies.sort(reverse=True, key=lambda x: x[0])
        return [movie for score, movie in scored_movies[:top_n]]

def display_movie_card(movie, show_details=False, recommender=None, show_like_button=False, current_mode=None):
    """Display a movie card with poster and information"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster_path = movie.get('poster_path')
        if poster_path:
            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            st.image(poster_url, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
    
    with col2:
        st.markdown(f"### {movie.get('title', 'Unknown Title')}")
        
        # Rating
        rating = movie.get('vote_average', 0)
        st.markdown(f"⭐ **Rating:** {rating}/10 ({movie.get('vote_count', 0)} votes)")
        
        # Release date
        release_date = movie.get('release_date', 'N/A')
        st.markdown(f"📅 **Release Date:** {release_date}")
        
        # Overview
        overview = movie.get('overview', 'No overview available.')
        st.markdown(f"**Overview:** {overview[:200]}..." if len(overview) > 200 else f"**Overview:** {overview}")
        
        # Genres (if available in details)
        if 'genres' in movie:
            genres = ', '.join([g['name'] for g in movie['genres']])
            st.markdown(f"🎭 **Genres:** {genres}")
        
        # Popularity
        popularity = movie.get('popularity', 0)
        st.markdown(f"🔥 **Popularity:** {popularity:.1f}")
        
        # Action buttons
        if show_details or show_like_button:
            button_cols = st.columns([1, 1, 2])
            
            # Like/Unlike button
            if show_like_button and recommender:
                with button_cols[0]:
                    is_liked = recommender.is_movie_liked(movie['id'])
                    if is_liked:
                        if st.button(f"💔 Unlike", key=f"unlike_{movie['id']}"):
                            recommender.unlike_movie(movie['id'])
                            st.rerun()
                    else:
                        if st.button(f"❤️ Like", key=f"like_{movie['id']}"):
                            recommender.like_movie(movie['id'], movie, mode=current_mode)
                            st.success("❤️ Liked! This will improve your recommendations in Search Movies mode.")
                            st.rerun()
            
            # Get Recommendations button
            if show_details:
                with button_cols[1]:
                    if st.button(f"🎯 Recommend", key=f"rec_{movie['id']}"):
                        st.session_state['selected_movie_id'] = movie['id']
                        st.session_state['show_recommendations'] = True
                        st.rerun()

def main():
    # Anchor point for scrolling to top
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    st.title("🎬 AI-Powered Movie Recommendation System")
    st.markdown("### Discover your next favorite movie with live recommendations!")
    
    # Check API key
    if not TMDB_API_KEY:
        st.error("⚠️ TMDb API Key not found! Please set TMDB_API_KEY in your .env file")
        st.info("""
        **How to get your TMDb API Key:**
        1. Go to https://www.themoviedb.org/
        2. Create a free account
        3. Go to Settings > API
        4. Request an API key (choose "Developer" option)
        5. Copy your API key and add it to .env file as: TMDB_API_KEY=your_key_here
        """)
        return
    
    # Initialize recommender
    recommender = MovieRecommender(TMDB_API_KEY)
    
    # Sidebar
    st.sidebar.title("🎯 Filters & Options")
    
    # Search or Browse
    mode = st.sidebar.radio("Mode", ["🔍 Search Movies", "🌟 Browse Trending", "⭐ Latest Movies", "🎭 Browse by Genre", "🌍 Browse by Language", "📅 Browse by Year", "🎤 Dubbed Movies", "🎯 Advanced Filters"])
    
    # Show search history in sidebar for Search Movies mode
    if mode == "🔍 Search Movies":
        if recommender.liked_movies:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### ❤️ Liked Movies")
            st.sidebar.caption(f"Total: {len(recommender.liked_movies)} movies from all modes")
            
            # Show recent liked movies
            st.sidebar.markdown("**Recent Likes:**")
            sorted_likes = sorted(recommender.liked_movies.items(), 
                                key=lambda x: x[1].get('timestamp', datetime.now()), 
                                reverse=True)
            
            for idx, (movie_id, data) in enumerate(sorted_likes[:5]):
                movie = data.get('movie_data', {})
                title = movie.get('title', 'Unknown')
                movie_mode = data.get('mode', 'Unknown')
                timestamp = data.get('timestamp', datetime.now())
                
                # Mode emoji mapping
                mode_emoji = {
                    '🔍 Search Movies': '🔍',
                    '🌟 Browse Trending': '🌟',
                    '⭐ Latest Movies': '⭐',
                    '🎭 Browse by Genre': '🎭',
                    '🌍 Browse by Language': '🌍',
                    '📅 Browse by Year': '📅',
                    '🎤 Dubbed Movies': '🎤',
                    '🎯 Advanced Filters': '🎯'
                }
                emoji = mode_emoji.get(movie_mode, '❤️')
                
                st.sidebar.text(f"{emoji} {title[:22]}...")
                st.sidebar.caption(f"   {timestamp.strftime('%m/%d %H:%M')}")
            
            if st.sidebar.button("❤️ View All Liked"):
                st.session_state['show_liked_movies'] = True
                st.rerun()
        
        if recommender.user_search_history:
            st.sidebar.markdown("---")
            
            # Show preferred languages
            all_languages = []
            for entry in recommender.user_search_history:
                all_languages.extend(entry.get('languages', []))
            
            if all_languages:
                language_frequency = {}
                for lang in all_languages:
                    if lang:
                        language_frequency[lang] = language_frequency.get(lang, 0) + 1
                
                preferred_languages = [lang for lang, count in language_frequency.items() if count >= 2]
                if preferred_languages:
                    preferred_languages.sort(key=lambda x: language_frequency[x], reverse=True)
                    st.sidebar.markdown("### 🌍 Your Preferred Languages")
                    for lang in preferred_languages[:3]:
                        count = language_frequency[lang]
                        st.sidebar.text(f"• {lang.upper()} ({count} times)")
            
            st.sidebar.markdown("### 📜 Recent Searches")
            recent_searches = [entry for entry in reversed(recommender.user_search_history[-5:]) if entry['query'] != 'browsing']
            if recent_searches:
                for idx, entry in enumerate(recent_searches):
                    timestamp = entry['timestamp'].strftime("%m/%d %H:%M")
                    st.sidebar.text(f"{idx+1}. {entry['query'][:20]}...")
                    st.sidebar.caption(f"   {timestamp}")
            
            if st.sidebar.button("🗑️ Clear History"):
                recommender.user_search_history = []
                recommender.save_search_history()
                st.rerun()
    
    # Initialize session state
    if 'selected_movie_id' not in st.session_state:
        st.session_state['selected_movie_id'] = None
    if 'show_recommendations' not in st.session_state:
        st.session_state['show_recommendations'] = False
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1
    if 'previous_mode' not in st.session_state:
        st.session_state['previous_mode'] = mode
    
    # Reset page when mode changes
    if st.session_state['previous_mode'] != mode:
        st.session_state['current_page'] = 1
        st.session_state['previous_mode'] = mode
    
    # Main content area
    if mode == "🔍 Search Movies":
        st.subheader("🔍 Search Movies")
        
        # Initialize search page state
        if 'search_page' not in st.session_state:
            st.session_state['search_page'] = 1
        if 'previous_search_query' not in st.session_state:
            st.session_state['previous_search_query'] = ""
        
        # Search input at the top
        search_query = st.text_input("🔎 Search for a movie:", placeholder="Enter movie name...")
        
        # Reset page if search query changes
        if search_query != st.session_state['previous_search_query']:
            st.session_state['search_page'] = 1
            st.session_state['previous_search_query'] = search_query
        
        if search_query:
            st.session_state['search_active'] = True
            
            # Page navigation at top
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_search_top") and st.session_state['search_page'] > 1:
                    st.session_state['search_page'] -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['search_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_search_top"):
                    st.session_state['search_page'] += 1
                    st.rerun()
            
            with st.spinner("Searching movies..."):
                movies, total_pages = recommender.search_movies(search_query, page=st.session_state['search_page'])
            
            if movies:
                st.success(f"Found movies - Page {st.session_state['search_page']} of {total_pages}")
                
                # Track search, movie IDs, and languages (only on first page)
                if st.session_state['search_page'] == 1:
                    movie_ids = [movie['id'] for movie in movies]
                    languages = [movie.get('original_language') for movie in movies if movie.get('original_language')]
                    recommender.add_to_search_history(search_query, movie_ids, languages)
                
                for movie in movies:
                    with st.container():
                        st.markdown("---")
                        display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                
                # Bottom pagination
                st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="prev_search_bottom") and st.session_state['search_page'] > 1:
                        st.session_state['search_page'] -= 1
                        st.rerun()
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['search_page']} of {total_pages}</h3>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ➡️", key="next_search_bottom") and st.session_state['search_page'] < total_pages:
                        st.session_state['search_page'] += 1
                        st.rerun()
            else:
                st.warning("No movies found. Try a different search term.")
        else:
            st.session_state['search_active'] = False
            st.session_state['search_page'] = 1  # Reset page when search is cleared
            
            # Show liked movies section if requested
            if st.session_state.get('show_liked_movies', False):
                st.markdown("---")
                st.markdown("### ❤️ Your Liked Movies")
                
                if recommender.liked_movies:
                    # Header with back button and view options
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown(f"**Total: {len(recommender.liked_movies)} movies**")
                    with col2:
                        view_option = st.selectbox(
                            "View by:",
                            ["All Movies", "Group by Mode", "Recent First", "Oldest First"],
                            key="liked_view_option"
                        )
                    with col3:
                        if st.button("🔙 Back"):
                            st.session_state['show_liked_movies'] = False
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Group by mode
                    if view_option == "Group by Mode":
                        mode_groups = {}
                        for movie_id, data in recommender.liked_movies.items():
                            movie_mode = data.get('mode', 'Unknown')
                            if movie_mode not in mode_groups:
                                mode_groups[movie_mode] = []
                            mode_groups[movie_mode].append((movie_id, data))
                        
                        # Display grouped by mode
                        for movie_mode, movies in sorted(mode_groups.items()):
                            st.markdown(f"## {movie_mode}")
                            st.caption(f"{len(movies)} movies")
                            
                            for movie_id, data in movies:
                                movie = data.get('movie_data', {})
                                timestamp = data.get('timestamp', datetime.now())
                                
                                with st.container():
                                    st.markdown("---")
                                    # Show timestamp
                                    st.caption(f"Liked on: {timestamp.strftime('%B %d, %Y at %H:%M')}")
                                    display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                            
                            st.markdown("")  # Add spacing between groups
                    
                    # Sort by recent
                    elif view_option == "Recent First":
                        sorted_likes = sorted(recommender.liked_movies.items(), 
                                            key=lambda x: x[1].get('timestamp', datetime.now()), 
                                            reverse=True)
                        
                        for movie_id, data in sorted_likes:
                            movie = data.get('movie_data', {})
                            movie_mode = data.get('mode', 'Unknown')
                            timestamp = data.get('timestamp', datetime.now())
                            
                            with st.container():
                                st.markdown("---")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"Liked on: {timestamp.strftime('%B %d, %Y at %H:%M')}")
                                with col2:
                                    st.caption(f"From: {movie_mode}")
                                display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                    
                    # Sort by oldest
                    elif view_option == "Oldest First":
                        sorted_likes = sorted(recommender.liked_movies.items(), 
                                            key=lambda x: x[1].get('timestamp', datetime.now()))
                        
                        for movie_id, data in sorted_likes:
                            movie = data.get('movie_data', {})
                            movie_mode = data.get('mode', 'Unknown')
                            timestamp = data.get('timestamp', datetime.now())
                            
                            with st.container():
                                st.markdown("---")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"Liked on: {timestamp.strftime('%B %d, %Y at %H:%M')}")
                                with col2:
                                    st.caption(f"From: {movie_mode}")
                                display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                    
                    # All movies (default)
                    else:
                        for movie_id, data in recommender.liked_movies.items():
                            movie = data.get('movie_data', {})
                            movie_mode = data.get('mode', 'Unknown')
                            timestamp = data.get('timestamp', datetime.now())
                            
                            with st.container():
                                st.markdown("---")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"Liked on: {timestamp.strftime('%B %d, %Y at %H:%M')}")
                                with col2:
                                    st.caption(f"From: {movie_mode}")
                                display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                else:
                    st.info("You haven't liked any movies yet. Start liking movies to see them here!")
                    if st.button("🔙 Back"):
                        st.session_state['show_liked_movies'] = False
                        st.rerun()
            else:
                # Show personalized recommendations when no search query
                st.markdown("---")
                if recommender.liked_movies:
                    st.markdown("### 🎬 Recommended For You")
                    st.markdown("*Based on your liked movies and viewing history*")
                    
                    # Show recommendation info
                    with st.expander("ℹ️ How your recommendations are personalized"):
                        st.markdown(f"""
                        **Your Recommendation Profile:**
                        - ❤️ **{len(recommender.liked_movies)} Liked Movies** from all browsing modes
                        - 📜 **{len(recommender.user_search_history)} Search History** entries
                        
                        **We analyze:**
                        1. Genres from your liked movies (+8 points per match)
                        2. Languages you frequently search (+10 points)
                        3. Your viewing patterns and preferences
                        
                        **💡 Tip:** Like movies in any mode (Trending, Genre, Language, etc.) 
                        to improve your recommendations here!
                        """)
                elif recommender.user_search_history:
                    st.markdown("### 🎬 Recommended For You")
                    st.markdown("*Based on your viewing history*")
                    st.info("💡 **Tip:** Start liking movies to get even better recommendations!")
                else:
                    st.markdown("### 🔥 Trending Now")
                    st.markdown("*Popular movies this week*")
                    st.info("💡 **Tip:** Search and like movies to build your personalized recommendation profile!")
                
                with st.spinner("Loading recommendations..."):
                    recommended_movies = recommender.get_recommendations_from_history(top_n=15)
                
                if recommended_movies:
                    # Display recommendations in a grid-like format
                    for idx, movie in enumerate(recommended_movies):
                        with st.container():
                            st.markdown("---")
                            display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
    
    elif mode == "🌟 Browse Trending":
        st.subheader("🔥 Trending Movies This Week")
        
        # Optional filters in sidebar
        st.sidebar.markdown("### 🎯 Trending Filters (Optional)")
        
        # Language filter
        languages = {
            "All Languages": None,
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Turkish": "tr",
            "Thai": "th",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Bengali": "bn",
            "Marathi": "mr"
        }
        
        selected_trending_lang = st.sidebar.selectbox(
            "Language",
            options=list(languages.keys()),
            help="Filter trending movies by language"
        )
        
        # Year filter
        current_year = datetime.now().year
        years = ["All Years"] + list(range(current_year, current_year - 10, -1))
        
        selected_trending_year = st.sidebar.selectbox(
            "Year",
            options=years,
            help="Filter trending movies by release year"
        )
        
        # Get filter values
        trending_lang_code = languages.get(selected_trending_lang)
        trending_year = selected_trending_year if selected_trending_year != "All Years" else None
        
        # Reset page when filters change
        trending_filter_key = f"{selected_trending_lang}_{selected_trending_year}"
        if 'previous_trending_filter' not in st.session_state:
            st.session_state['previous_trending_filter'] = trending_filter_key
        if st.session_state['previous_trending_filter'] != trending_filter_key:
            st.session_state['current_page'] = 1
            st.session_state['previous_trending_filter'] = trending_filter_key
        
        # Display active filters
        active_filters = []
        if selected_trending_lang != "All Languages":
            active_filters.append(f"🌍 {selected_trending_lang}")
        if selected_trending_year != "All Years":
            active_filters.append(f"📅 {selected_trending_year}")
        
        if active_filters:
            st.info(f"**Active Filters:** {' + '.join(active_filters)}")
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", key="prev_trending") and st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="next_trending"):
                st.session_state['current_page'] += 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        
        with st.spinner("Loading trending movies..."):
            movies = recommender.fetch_trending_movies(page=st.session_state['current_page'], language_code=trending_lang_code, year=trending_year)
        
        if movies:
            for movie in movies:
                with st.container():
                    st.markdown("---")
                    display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
            
            # Bottom pagination
            st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_trending_bottom") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_trending_bottom"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
        else:
            st.warning("No more movies available.")
            if st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
    
    elif mode == "⭐ Latest Movies":
        st.subheader("⭐ Latest Released Movies")
        st.markdown("**Now playing in theaters and recently released**")
        
        # Language filter in sidebar
        st.sidebar.markdown("### 🌍 Language Filter (Optional)")
        
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Turkish": "tr",
            "Thai": "th",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Bengali": "bn",
            "Marathi": "mr"
        }
        
        selected_languages = st.sidebar.multiselect(
            "Select Language(s)",
            options=list(languages.keys()),
            default=[],
            help="Shows movies in original language + dubbed versions available in selected languages"
        )
        
        include_dubbed = st.sidebar.checkbox(
            "Include dubbed movies",
            value=True,
            help="When enabled, shows movies available in selected languages (original + dubbed)"
        )
        
        # Get language codes
        language_codes = [languages[lang] for lang in selected_languages] if selected_languages else None
        
        # Reset page when language filter changes
        lang_filter_key = ','.join(selected_languages) if selected_languages else 'all'
        if 'previous_latest_lang' not in st.session_state:
            st.session_state['previous_latest_lang'] = lang_filter_key
        if st.session_state['previous_latest_lang'] != lang_filter_key:
            st.session_state['current_page'] = 1
            st.session_state['previous_latest_lang'] = lang_filter_key
        
        # Display active language filters
        if selected_languages:
            st.info(f"🌍 **Filtering by:** {', '.join(selected_languages)}")
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", key="prev_latest") and st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="next_latest"):
                st.session_state['current_page'] += 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        
        with st.spinner("Loading latest movies..."):
            movies = recommender.get_latest_movies(page=st.session_state['current_page'], language_codes=language_codes, include_dubbed=include_dubbed)
        
        if movies:
            for movie in movies:
                with st.container():
                    st.markdown("---")
                    display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
            
            # Bottom pagination
            st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_latest_bottom") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_latest_bottom"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
        else:
            st.warning("No more movies available.")
            if st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
    
    elif mode == "🎭 Browse by Genre":
        genres = recommender.get_genres()
        
        if genres:
            genre_dict = {g['name']: g['id'] for g in genres}
            selected_genre = st.sidebar.selectbox("Select Genre", list(genre_dict.keys()))
            
            # Reset page when genre changes
            if 'previous_genre' not in st.session_state:
                st.session_state['previous_genre'] = selected_genre
            if st.session_state['previous_genre'] != selected_genre:
                st.session_state['current_page'] = 1
                st.session_state['previous_genre'] = selected_genre
            
            if selected_genre:
                st.subheader(f"🎬 {selected_genre} Movies")
                
                # Page navigation
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="prev_genre") and st.session_state['current_page'] > 1:
                        st.session_state['current_page'] -= 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ➡️", key="next_genre"):
                        st.session_state['current_page'] += 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
                
                with st.spinner(f"Loading {selected_genre} movies..."):
                    movies = recommender.get_movie_by_genre(genre_dict[selected_genre], page=st.session_state['current_page'])
                
                if movies:
                    for movie in movies:
                        with st.container():
                            st.markdown("---")
                            display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
                    
                    # Bottom pagination
                    st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        if st.button("⬅️ Previous", key="prev_genre_bottom") and st.session_state['current_page'] > 1:
                            st.session_state['current_page'] -= 1
                            st.session_state['scroll_to_top'] = True
                            st.rerun()
                    with col2:
                        st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
                    with col3:
                        if st.button("Next ➡️", key="next_genre_bottom"):
                            st.session_state['current_page'] += 1
                            st.session_state['scroll_to_top'] = True
                            st.rerun()
                else:
                    st.warning("No more movies available.")
                    if st.session_state['current_page'] > 1:
                        st.session_state['current_page'] -= 1
    
    elif mode == "🌍 Browse by Language":
        # Popular languages with their ISO 639-1 codes
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Turkish": "tr",
            "Thai": "th",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Bengali": "bn",
            "Marathi": "mr"
        }
        
        selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()))
        
        # Reset page when language changes
        if 'previous_language' not in st.session_state:
            st.session_state['previous_language'] = selected_language
        if st.session_state['previous_language'] != selected_language:
            st.session_state['current_page'] = 1
            st.session_state['previous_language'] = selected_language
        
        if selected_language:
            st.subheader(f"🌍 {selected_language} Movies")
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_language") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_language"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            
            with st.spinner(f"Loading {selected_language} movies..."):
                movies = recommender.get_movie_by_language(languages[selected_language], page=st.session_state['current_page'])
            
            if movies:
                for movie in movies:
                    with st.container():
                        st.markdown("---")
                        display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True)
                
                # Bottom pagination
                st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="prev_language_bottom") and st.session_state['current_page'] > 1:
                        st.session_state['current_page'] -= 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ➡️", key="next_language_bottom"):
                        st.session_state['current_page'] += 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
            else:
                st.warning("No more movies available.")
                if st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
    
    elif mode == "📅 Browse by Year":
        # Generate year range from 1900 to current year
        current_year = datetime.now().year
        years = list(range(current_year, 1899, -1))  # Current year down to 1900
        
        selected_year = st.sidebar.selectbox("Select Year", years)
        
        # Reset page when year changes
        if 'previous_year' not in st.session_state:
            st.session_state['previous_year'] = selected_year
        if st.session_state['previous_year'] != selected_year:
            st.session_state['current_page'] = 1
            st.session_state['previous_year'] = selected_year
        
        if selected_year:
            st.subheader(f"📅 Movies from {selected_year}")
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_year") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_year"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            
            with st.spinner(f"Loading movies from {selected_year}..."):
                movies = recommender.get_movie_by_year(selected_year, page=st.session_state['current_page'])
            
            if movies:
                for movie in movies:
                    with st.container():
                        st.markdown("---")
                        display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True)
                
                # Bottom pagination
                st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="prev_year_bottom") and st.session_state['current_page'] > 1:
                        st.session_state['current_page'] -= 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ➡️", key="next_year_bottom"):
                        st.session_state['current_page'] += 1
                        st.session_state['scroll_to_top'] = True
                        st.rerun()
            else:
                st.warning("No more movies available.")
                if st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
    
    elif mode == "🎤 Dubbed Movies":
        st.subheader("🎤 Browse Dubbed Movies")
        st.markdown("**Find movies available in your preferred language (dubbed/translated)**")
        
        # Language selection
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Turkish": "tr",
            "Thai": "th",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Bengali": "bn",
            "Marathi": "mr"
        }
        
        st.sidebar.markdown("### 🎤 Dubbed Language")
        selected_dubbed_lang = st.sidebar.selectbox(
            "Select dubbed language",
            options=list(languages.keys()),
            help="Shows movies available/dubbed in this language (excluding originals)"
        )
        
        target_lang_code = languages[selected_dubbed_lang]
        
        # Reset page when language changes
        if 'previous_dubbed_lang' not in st.session_state:
            st.session_state['previous_dubbed_lang'] = selected_dubbed_lang
        if st.session_state['previous_dubbed_lang'] != selected_dubbed_lang:
            st.session_state['current_page'] = 1
            st.session_state['previous_dubbed_lang'] = selected_dubbed_lang
        
        # Option to include or exclude original language movies
        exclude_originals = st.sidebar.checkbox(
            "Exclude movies originally in this language",
            value=True,
            help="When enabled, only shows foreign movies likely dubbed in your selected language"
        )
        
        st.info(f"🎤 **Showing popular movies for dubbing in:** {selected_dubbed_lang}")
        if exclude_originals:
            st.info("✅ Excluding movies originally made in {}".format(selected_dubbed_lang))
        st.markdown("""
        **Note:** This mode shows popular movies that are commonly dubbed. 
        The actual availability of dubbed versions depends on your region and streaming platforms.
        """)
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", key="prev_dubbed") and st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="next_dubbed"):
                st.session_state['current_page'] += 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        
        with st.spinner(f"Loading popular movies for {selected_dubbed_lang} dubbing..."):
            movies = recommender.get_dubbed_movies(target_lang_code, exclude_original=exclude_originals, page=st.session_state['current_page'])
        
        if movies:
            st.success(f"Found {len(movies)} dubbed movies on this page")
            for movie in movies:
                with st.container():
                    st.markdown("---")
                    display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
            
            # Bottom pagination
            st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_dubbed_bottom") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_dubbed_bottom"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
        else:
            st.warning(f"⚠️ No dubbed movies found in {selected_dubbed_lang}. Try a different language or page.")
            if st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
    
    elif mode == "🎯 Advanced Filters":
        st.subheader("🎯 Advanced Filters - Combine Multiple Criteria")
        st.markdown("**Select any combination of filters below:**")
        
        # Get genres
        genres = recommender.get_genres()
        genre_dict = {g['name']: g['id'] for g in genres} if genres else {}
        
        # Languages
        languages = {
            "Any Language": None,
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Turkish": "tr",
            "Thai": "th",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Bengali": "bn",
            "Marathi": "mr"
        }
        
        # Year range
        current_year = datetime.now().year
        years = ["Any Year"] + list(range(current_year, 1899, -1))
        
        # Filter selections in sidebar
        st.sidebar.markdown("### 🎯 Filter Options")
        selected_genre = st.sidebar.selectbox("Genre", ["Any Genre"] + list(genre_dict.keys()))
        selected_language = st.sidebar.selectbox("Language", list(languages.keys()))
        selected_year = st.sidebar.selectbox("Year", years)
        
        # Get filter values
        genre_id = genre_dict.get(selected_genre) if selected_genre != "Any Genre" else None
        language_code = languages.get(selected_language)
        year_value = selected_year if selected_year != "Any Year" else None
        
        # Reset page when filters change
        filter_key = f"{selected_genre}_{selected_language}_{selected_year}"
        if 'previous_filter' not in st.session_state:
            st.session_state['previous_filter'] = filter_key
        if st.session_state['previous_filter'] != filter_key:
            st.session_state['current_page'] = 1
            st.session_state['previous_filter'] = filter_key
        
        # Display active filters
        active_filters = []
        if selected_genre != "Any Genre":
            active_filters.append(f"🎭 {selected_genre}")
        if selected_language != "Any Language":
            active_filters.append(f"🌍 {selected_language}")
        if selected_year != "Any Year":
            active_filters.append(f"📅 {selected_year}")
        
        if active_filters:
            st.info(f"**Active Filters:** {' + '.join(active_filters)}")
        else:
            st.info("ℹ️ **No filters selected** - Showing popular movies")
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", key="prev_advanced") and st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="next_advanced"):
                st.session_state['current_page'] += 1
                st.session_state['scroll_to_top'] = True
                st.rerun()
        
        with st.spinner("Loading movies with your filters..."):
            movies = recommender.get_movies_with_filters(
                genre_id=genre_id,
                language_code=language_code,
                year=year_value,
                page=st.session_state['current_page']
            )
        
        if movies:
            st.success(f"Found {len(movies)} movies on this page")
            for movie in movies:
                with st.container():
                    st.markdown("---")
                    display_movie_card(movie, show_details=True, recommender=recommender, show_like_button=True, current_mode=mode)
            
            # Bottom pagination
            st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_advanced_bottom") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state['current_page']}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_advanced_bottom"):
                    st.session_state['current_page'] += 1
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
        else:
            st.warning("⚠️ No movies found with the selected filters. Try different combinations.")
            if st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
    
    # Show recommendations if a movie is selected
    if st.session_state.get('show_recommendations') and st.session_state.get('selected_movie_id'):
        st.markdown("---")
        st.header("🎯 Recommended Movies For You")
        
        # Initialize recommendation page state
        if 'recommendation_page' not in st.session_state:
            st.session_state['recommendation_page'] = 1
        if 'all_recommendations' not in st.session_state:
            st.session_state['all_recommendations'] = []
        
        movie_id = st.session_state['selected_movie_id']
        
        # Generate recommendations only once
        if not st.session_state['all_recommendations'] or st.session_state.get('last_recommended_movie_id') != movie_id:
            with st.spinner("Generating personalized recommendations..."):
                # Get more movies for better recommendations
                base_movies = []
                for page in range(1, 6):  # Get 5 pages (100 movies)
                    base_movies.extend(recommender.fetch_popular_movies(page=page))
                
                # Get more recommendations
                all_recs = recommender.get_content_based_recommendations(movie_id, base_movies, top_n=50)
                st.session_state['all_recommendations'] = all_recs
                st.session_state['last_recommended_movie_id'] = movie_id
                st.session_state['recommendation_page'] = 1
        
        recommendations = st.session_state['all_recommendations']
        
        if recommendations:
            # Pagination settings
            items_per_page = 10
            total_pages = (len(recommendations) + items_per_page - 1) // items_per_page
            current_page = st.session_state['recommendation_page']
            
            # Calculate start and end indices
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(recommendations))
            page_recommendations = recommendations[start_idx:end_idx]
            
            st.success(f"Found {len(recommendations)} similar movies!")
            
            # Top pagination
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_rec_top") and current_page > 1:
                    st.session_state['recommendation_page'] -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {current_page} of {total_pages}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_rec_top") and current_page < total_pages:
                    st.session_state['recommendation_page'] += 1
                    st.rerun()
            
            # Display recommendations for current page
            for rec_movie in page_recommendations:
                with st.container():
                    st.markdown("---")
                    display_movie_card(rec_movie, show_details=False, recommender=recommender, show_like_button=True, current_mode="🎯 Recommendations")
                    
                    # OpenCV poster analysis
                    poster_path = rec_movie.get('poster_path')
                    if poster_path:
                        with st.expander("🎨 Visual Analysis (OpenCV)"):
                            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                            analysis = recommender.process_poster_with_opencv(poster_url)
                            
                            if analysis:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Brightness", f"{analysis['brightness']:.1f}")
                                with col2:
                                    st.metric("Complexity", f"{analysis['complexity']:.3f}")
                                with col3:
                                    color = analysis['dominant_color']
                                    st.markdown(f"**Dominant Color:** RGB{tuple(color)}")
            
            # Bottom pagination
            st.markdown("<div class='pagination-bottom'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_rec_bottom") and current_page > 1:
                    st.session_state['recommendation_page'] -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>Page {current_page} of {total_pages}</h3>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_rec_bottom") and current_page < total_pages:
                    st.session_state['recommendation_page'] += 1
                    st.rerun()
        else:
            st.warning("No recommendations found. Try selecting a different movie.")
        
        if st.button("🔙 Back to Browse"):
            st.session_state['show_recommendations'] = False
            st.session_state['selected_movie_id'] = None
            st.session_state['all_recommendations'] = []
            st.session_state['recommendation_page'] = 1
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This movie recommender uses:
    - **TMDb API** for live movie data
    - **Content-based filtering** for recommendations
    - **OpenCV** for poster image analysis
    - **TF-IDF & Cosine Similarity** for matching
    """)

if __name__ == "__main__":
    main()
