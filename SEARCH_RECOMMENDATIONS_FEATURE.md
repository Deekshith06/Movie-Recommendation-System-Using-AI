# Netflix-Style Recommendations in Search Movies Mode

## Overview
The Search Movies mode now includes personalized movie recommendations similar to Netflix's "Top Picks for You" feature. Movies are recommended based on your search and viewing history.

## Features

### 1. **Personalized Recommendations**
- When you first open Search Movies mode (before searching), you'll see recommended movies
- If you have search history: Shows "🎬 Recommended For You" based on your viewing patterns
- If no history: Shows "🔥 Trending Now" with popular movies

### 2. **Smart Recommendation Algorithm**
The system analyzes your search history to recommend movies by:
- **Genre Matching**: Identifies your preferred genres from past searches
- **Popularity Score**: Prioritizes highly-rated and popular movies
- **Recent Releases**: Gives bonus points to movies released in the last 2 years
- **Diversity**: Combines genre-based recommendations with trending movies

### 3. **Search History Tracking**
- Automatically tracks all your searches
- Stores movie IDs of movies you've viewed
- Keeps last 50 searches for optimal performance
- Persistent storage using `search_history.pkl`

### 4. **Sidebar Features**
- **Recent Searches**: View your last 5 searches with timestamps
- **Clear History**: Button to reset your search history and start fresh

### 5. **Interactive Elements**
- **View Details Button**: Click to show interest in a movie (tracked for better recommendations)
- **Get Recommendations**: Get similar movies based on any movie you like

## How It Works

### First Time Users
1. Open Search Movies mode
2. See "🔥 Trending Now" with popular movies
3. Browse and search for movies
4. Your searches are tracked automatically

### Returning Users
1. Open Search Movies mode
2. See "🎬 Recommended For You" with personalized picks
3. Recommendations improve as you search more
4. Based on genres and movies you've shown interest in

### Recommendation Scoring
Each movie gets a score based on:
- **Genre Match**: +5 points per matching genre
- **Popularity**: +0.02 × popularity score
- **Rating**: +1.0 × vote average
- **Recent Release**: +3 points if released within 2 years

Movies are sorted by score and top 15 are displayed.

## Data Storage
- **File**: `search_history.pkl`
- **Location**: Project root directory
- **Content**: Search queries, movie IDs, timestamps
- **Limit**: Last 50 searches (automatic cleanup)

## Privacy
- All data is stored locally on your machine
- No data is sent to external servers (except TMDb API calls)
- You can clear your history anytime using the "Clear History" button

## Example User Journey

### Day 1
```
User searches: "Avengers"
→ System tracks: Action, Adventure, Sci-Fi genres
→ Stores: Movie IDs from search results
```

### Day 2
```
User opens Search Movies
→ System shows: Action/Adventure movies + Trending
→ Recommendations include: Similar superhero movies, popular action films
```

### Day 3
```
User searches: "The Godfather"
→ System updates: Adds Crime, Drama genres to preferences
→ Next recommendations: Mix of Action and Drama movies
```

## Technical Details

### Methods Added
1. `load_search_history()` - Loads saved search history
2. `save_search_history()` - Persists search history to disk
3. `add_to_search_history(query, movie_ids)` - Tracks user searches
4. `get_recommendations_from_history(top_n)` - Generates personalized recommendations

### Algorithm Flow
```
1. Load user search history
2. Extract movie IDs from history
3. Fetch movie details for recent movies
4. Identify preferred genres and keywords
5. Fetch movies from preferred genres
6. Add trending movies for diversity
7. Remove duplicates and already-viewed movies
8. Score each movie based on multiple factors
9. Sort by score and return top N
```

## Benefits
- **Personalized Experience**: Like Netflix, recommendations improve over time
- **Discovery**: Find movies you might not have searched for
- **Convenience**: No need to search if recommendations are good
- **Smart**: Balances your preferences with popular/trending content
