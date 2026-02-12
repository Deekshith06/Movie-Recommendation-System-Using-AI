import streamlit as st
import pickle
import pandas as pd
import requests

# ================== CONFIG & UI THEME ==================
# Using your OMDB Key
try:
    API_KEY = st.secrets["OMDB_API_KEY"]
except KeyError:
    st.error("OMDB API Key not found in secrets.toml.")
    st.stop()

st.set_page_config(page_title="NextWatch", layout="wide")

st.markdown("""
<style>
    /* Global Styles */
    .main { 
        background-color: #0d1117; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Clean Grid Spacing */
    .stColumn {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* Movie Card - Poster Only */
    .movie-card {
        background-color: transparent;
        border: none;
        box-shadow: none;
        overflow: visible;
        height: auto;
        display: flex;
        flex-direction: column;
        margin-bottom: 2rem;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    
    /* Hover Effect - Scale Image Only */
    .movie-card:hover {
        transform: translateY(-5px);
    }
    
    /* Poster Image */
    .movie-img {
        width: 100%;
        height: 240px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        object-fit: cover;
        object-position: top center;
        opacity: 0.95;
        transition: opacity 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
        display: block;
    }
    
    .movie-card:hover .movie-img {
        opacity: 1;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5);
    }
    
    /* Title Section - Below Poster */
    .movie-title {
        height: auto;
        min-height: 40px;
        padding: 10px 5px 0 5px;
        text-align: center;
        width: 100%;
        
        /* Typography */
        color: #e6edf3;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.02em;
        line-height: 1.4;
        
        /* Background Removed */
        background: transparent;
        border: none;
        
        /* Truncation */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Remove Link Decoration */
    a { text-decoration: none !important; color: inherit !important; }

    /* Button Styling */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
    }
    
    .detail-container { 
        background: #161b22; 
        padding: 2rem; 
        border-radius: 12px; 
        border: 1px solid #30363d;
        margin-bottom: 2rem; 
    }
</style>
""", unsafe_allow_html=True)


# ================== DATA LOADING (OPTIMIZED) ==================
@st.cache_resource  # This prevents the app from reloading data on every click
def load_assets():
    # UPDATED PATHS for Pro Structure
    movies = pickle.load(open("data/moviess.pkl", "rb"))
    
    # Handle split similarity file for GitHub compatibility
    import os
    import numpy as np
    
    if os.path.exists("data/similarities.pkl"):
        similarity = pickle.load(open("data/similarities.pkl", "rb"))
    elif os.path.exists("data/similarities_part1.pkl"):
        # Load parts and combine
        part1 = pickle.load(open("data/similarities_part1.pkl", "rb"))
        part2 = pickle.load(open("data/similarities_part2.pkl", "rb"))
        # Concatenate using numpy (assuming it's a list or array)
        if isinstance(part1, list):
             similarity = part1 + part2
        else:
             similarity = np.concatenate((part1, part2), axis=0)
    else:
        st.error("Similarity data not found (checked .pkl and split parts).")
        st.stop()
        
    return movies, similarity


try:
    movies, similarity = load_assets()
except FileNotFoundError:
    st.error("Data files not found in 'data/' directory. Please check file structure.")
    st.stop()


# ================== API & LOGIC ==================
def fetch_movie_details(title):
    """Fetches full movie metadata and poster from OMDB"""
    url = f"https://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if data.get("Response") == "True":
            return data
    except:
        pass
    return None


def recommend(movie_name, num_movies):
    index = movies[movies["title"] == movie_name].index[0]
    distances = similarity[index]

    # ADVANCED LOGIC: Get 30 similar movies first, then sort by rating
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:31]

    recommendations = []
    for i in movie_list:
        m_data = movies.iloc[i[0]]
        recommendations.append({
            'title': m_data.title,
            'rating': m_data.get('vote_average', 0)
        })

    # Sort the similar movies by their ratings (Hybrid approach)
    top_picks = sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:num_movies]

    names = []
    posters = []
    for m in top_picks:
        details = fetch_movie_details(m['title'])
        names.append(m['title'])
        poster = details.get("Poster") if details else None
        if poster in [None, "N/A", ""]:
            poster = "https://via.placeholder.com/300x450?text=No+Image"
        posters.append(poster)

    return names, posters


import urllib.parse

# ================== APP STRUCTURE ==================
# Handle URL parameters for direct linking
if "movie" in st.query_params:
    url_movie = st.query_params["movie"]
    # Ensure it's a valid movie
    if url_movie in movies["title"].values:
        if 'selected_movie_name' not in st.session_state or st.session_state.selected_movie_name != url_movie:
            st.session_state.selected_movie_name = url_movie
            st.session_state.movie_selectbox = url_movie

if 'trending_offset' not in st.session_state:
    st.session_state.trending_offset = 0

if 'selected_movie_name' not in st.session_state:
    st.session_state.selected_movie_name = None

def set_movie(movie_title):
    st.session_state.selected_movie_name = movie_title
    st.session_state.movie_selectbox = movie_title

def go_back():
    st.session_state.selected_movie_name = None
    st.query_params.clear()

st.title("NextWatch")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("Settings")
if st.session_state.selected_movie_name:
    if st.sidebar.button("Back to Trending", use_container_width=True):
        go_back()
        st.rerun()
        
num_rec = st.sidebar.number_input("Number of Recommendations", min_value=1, max_value=30, value=5)

# Centered Search Bar
col_spacer1, col_search, col_spacer2 = st.columns([1, 2, 1])
with col_search:
    # Safely get index
    try:
        val = st.session_state.get("selected_movie_name")
        default_index = list(movies["title"].values).index(val) if val in movies["title"].values else None
    except:
        default_index = None

    selected_movie = st.selectbox(
        "Type to search for a movie:",
        movies["title"].values,
        index=default_index,
        placeholder="Start typing a movie name...",
        key="movie_selectbox" 
    )

# Sync selection: If user changes selectbox, update state. If state changed by button, selectbox updates via reruns.
# Note: With URL params processing at top, this sync is still useful for manual selectbox changes.
if selected_movie != st.session_state.selected_movie_name:
     st.session_state.selected_movie_name = selected_movie

# Also update query param when selection changes (optional but good for sharing)
if selected_movie:
    st.query_params["movie"] = selected_movie
else:
    if "movie" in st.query_params:
        del st.query_params["movie"]

if selected_movie:
    # 1. Show Details of the Searched Movie
    with st.status("Fetching data...", expanded=False):
        current_info = fetch_movie_details(selected_movie)

    if current_info:
        with st.container():
            st.markdown('<div class="detail-container">', unsafe_allow_html=True)
            col_a, col_b = st.columns([1, 3])
            with col_a:
                poster = current_info.get("Poster")
                if poster in [None, "N/A", ""]:
                    poster = "https://via.placeholder.com/300x450?text=No+Image"
                st.image(poster, use_container_width=True)
            with col_b:
                st.subheader(f"{selected_movie} ({current_info.get('Year')})")
                st.write(
                    f"Rating: {current_info.get('imdbRating')} | Runtime: {current_info.get('Runtime')} | Genre: {current_info.get('Genre')}")
                st.write(f"**Plot:** {current_info.get('Plot')}")
                st.write(f"**Director:** {current_info.get('Director')}")
                st.write(f"**Cast:** {current_info.get('Actors')}")
                
                # Trailer Link
                trailer_url = f"https://www.youtube.com/results?search_query={selected_movie} trailer"
                st.link_button("Watch Trailer", trailer_url)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Because you liked that, you might love:")

    # 2. Get and Show Recommendations
    names, posters = recommend(selected_movie, num_rec)
    
    # Grid display for recommendations (Row-based for alignment)
    for i in range(0, len(names), 5):
        row_cols = st.columns(5, gap="medium")
        for j, col in enumerate(row_cols):
            if i + j < len(names):
                with col:
                    poster_url = posters[i+j]
                    if poster_url in [None, "N/A", ""]:
                        poster_url = "https://via.placeholder.com/300x450?text=No+Image"
                        
                    title = names[i+j]
                    link = f"?movie={urllib.parse.quote(title)}"
                    st.markdown(f"""
                        <a href="{link}" target="_self" style="text-decoration: none; color: inherit;">
                            <div class="movie-card">
                                <img class="movie-img" src="{poster_url}" alt="{title}" onerror="this.onerror=null;this.src='https://via.placeholder.com/300x450?text=No+Image';">
                                <div class="movie-title">{title}</div>
                            </div>
                        </a>
                    """, unsafe_allow_html=True)

else:
    # Default: Show Trending Movies
    # Align header and button vertically
    col_t1, col_t2 = st.columns([5, 1], gap="small")
    with col_t1:
        st.markdown("<h3 style='margin-top: 5px; margin-bottom: 20px;'>Trending Now</h3>", unsafe_allow_html=True)
    with col_t2:
        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True):
            st.session_state.trending_offset += 10
            st.rerun()
    
    # Get top movies with offset
    if 'vote_average' in movies.columns:
        # Sort by rating and take a slice based on offset
        sorted_movies = movies.sort_values(by='vote_average', ascending=False)
        start_idx = st.session_state.trending_offset % (len(sorted_movies) - 10) # Loop around
        trending_movies = sorted_movies.iloc[start_idx : start_idx + 10]
    else:
        # Just take a slice
        start_idx = st.session_state.trending_offset % (len(movies) - 10)
        trending_movies = movies.iloc[start_idx : start_idx + 10]
        
    trending_titles = trending_movies['title'].tolist()
    
    # Grid display for trending (Row-based for alignment)
    for i in range(0, len(trending_titles), 5):
        row_cols = st.columns(5, gap="medium")
        for j, col in enumerate(row_cols):
            if i + j < len(trending_titles):
                with col:
                    title = trending_titles[i+j]
                    details = fetch_movie_details(title)
                    poster_url = details.get("Poster") if details else "https://via.placeholder.com/300x450"
                    
                    # Handle N/A posters explicitly
                    if poster_url in [None, "N/A", ""]:
                        poster_url = "https://via.placeholder.com/300x450?text=No+Image"
                    
                    link = f"?movie={urllib.parse.quote(title)}"
                    st.markdown(f"""
                        <a href="{link}" target="_self" style="text-decoration: none; color: inherit;">
                            <div class="movie-card">
                                <img class="movie-img" src="{poster_url}" alt="{title}" onerror="this.onerror=null;this.src='https://via.placeholder.com/300x450?text=No+Image';">
                                <div class="movie-title">{title}</div>
                            </div>
                        </a>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><div style='text-align: center; color: #8b949e; font-size: 0.8rem;'>Designed for Professional Portfolio | 2026</div>", 
            unsafe_allow_html=True)
