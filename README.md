# 🎬 Movie-Recommendation-System-Using-AI

A modern, intelligent movie recommendation system built with **Streamlit**, **OpenCV**, and **The Movie Database (TMDb) API**. Get personalized movie recommendations in real-time with advanced content-based filtering and visual poster analysis.

## ✨ Features

### Core Features
- 🔍 **Live Movie Search** - Search thousands of movies in real-time with pagination
- 🌟 **Trending Movies** - Browse weekly trending movies with optional language and year filters
- ⭐ **Latest Movies** - Discover recently released movies with multi-language filtering
- 🎭 **Genre-Based Browsing** - Filter movies by your favorite genres
- 🌍 **Language Browsing** - Explore movies in 20+ languages including English, Hindi, Tamil, Telugu, Malayalam, Kannada, and more
- 📅 **Year-Based Browsing** - Browse movies from any year (1900 to present)
- 🎤 **Dubbed Movies** - Find popular movies available dubbed in your preferred language
- 🎯 **Advanced Filters** - Combine genre, language, and year filters for precise results

### Personalization Features
- ❤️ **Like System** - Like movies across all browsing modes to build your preference profile
- 🎬 **Personalized Recommendations** - Netflix-style "Recommended For You" based on liked movies and viewing history
- 📜 **Search History Tracking** - Automatic tracking of searches and viewed movies (last 50 searches)
- 🌍 **Language Preference Learning** - System learns your preferred languages from viewing patterns
- 🎭 **Genre Preference Analysis** - Recommendations prioritize genres from your liked movies

### Technical Features
- 🎯 **Smart Recommendations** - Content-based filtering using TF-IDF and cosine similarity
- 🎨 **Visual Analysis** - OpenCV-powered poster analysis (dominant colors, brightness, complexity)
- ⚡ **Fast & Responsive** - Powered by TMDb API for instant results
- 💾 **Intelligent Caching** - Improved performance with local caching system
- 🔄 **Pagination** - Smooth navigation through large result sets
- 🎨 **Modern UI** - Beautiful, Netflix-inspired dark theme

## 🛠️ Technologies Used

- **Streamlit** - Web framework for the UI
- **OpenCV** - Image processing and visual analysis
- **TMDb API** - Live movie data and metadata
- **scikit-learn** - Machine learning for recommendations (TF-IDF, Cosine Similarity)
- **Pandas & NumPy** - Data manipulation
- **Python-dotenv** - Environment variable management

## 📋 Prerequisites

- Python 3.8 or higher
- TMDb API Key (free)

## 🚀 Installation & Setup

### 1. Navigate to the Project Directory

```bash
cd "Movie-Recommendation-System-Using-AI"
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Install Dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

**Note for macOS users:** If you encounter an "externally-managed-environment" error, you must use a virtual environment (as shown above) rather than installing packages system-wide.

### 4. Get Your TMDb API Key

1. Go to [The Movie Database (TMDb)](https://www.themoviedb.org/)
2. Create a free account
3. Navigate to **Settings** → **API**
4. Request an API key (select "Developer" option)
5. Fill out the form (you can use "Personal" for application type)
6. Copy your API key

### 5. Configure Environment Variables

Create a `.env` file in the project directory:

```bash
nano.env
```

Edit the `.env` file and add your TMDb API key:

```
TMDB_API_KEY=your_actual_api_key_here
```
After Click
Ctrl+x -> y -> Enter 

### 6. Run the Application

```bash
.venv/bin/streamlit run movie_recommender.py
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate
streamlit run movie_recommender.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## 📖 How to Use

### Search for Movies
1. Select **"🔍 Search Movies"** mode in the sidebar
2. Enter a movie name in the search box
3. Browse through paginated results
4. Click **❤️ Like** to add movies to your favorites
5. Click **🎯 Recommend** to get similar movie suggestions
6. Your searches are automatically tracked to improve recommendations

### Personalized Recommendations
1. In **"🔍 Search Movies"** mode, leave the search box empty
2. If you have liked movies or search history:
   - See **"🎬 Recommended For You"** with personalized picks
   - Based on your liked movies (highest priority) and viewing patterns
3. First-time users see **"🔥 Trending Now"** until they build a profile
4. View your liked movies by clicking **"❤️ View All Liked"** in the sidebar

### Browse Trending Movies
1. Select **"🌟 Browse Trending"** mode
2. Optional: Filter by language and/or year in the sidebar
3. Scroll through this week's trending movies
4. Like movies to improve your recommendations

### Browse Latest Movies
1. Select **"⭐ Latest Movies"** mode
2. See recently released movies sorted by date
3. Optional: Filter by one or more languages
4. Toggle "Include dubbed movies" for broader results

### Browse by Genre
1. Select **"🎭 Browse by Genre"** mode
2. Choose a genre from the dropdown (Action, Comedy, Drama, etc.)
3. Explore popular movies in that genre
4. Like movies to add them to your profile

### Browse by Language
1. Select **"🌍 Browse by Language"** mode
2. Choose from 20+ languages (English, Hindi, Tamil, Telugu, Malayalam, Kannada, etc.)
3. Discover movies in your preferred language
4. System learns your language preferences over time

### Browse by Year
1. Select **"📅 Browse by Year"** mode
2. Choose any year from 1900 to present
3. Explore movies from that specific year

### Dubbed Movies
1. Select **"🎤 Dubbed Movies"** mode
2. Choose your preferred language
3. Find popular movies likely available dubbed in that language
4. Toggle "Exclude original language movies" to focus on foreign films

### Advanced Filters
1. Select **"🎯 Advanced Filters"** mode
2. Combine multiple filters:
   - Genre (Action, Drama, Comedy, etc.)
   - Language (20+ options)
   - Year (any year from 1900 to present)
3. Get precise results matching all your criteria

### View Visual Analysis
- Expand the **"🎨 Visual Analysis (OpenCV)"** section under recommended movies
- See poster brightness, complexity, and dominant colors
- Powered by OpenCV's image processing algorithms

## 🧠 How It Works

### Personalized Recommendation System
The system creates a unique profile for each user:

#### Profile Building
1. **Liked Movies** (Highest Priority)
   - Tracks movies you like across all browsing modes
   - Extracts genres, languages, and themes from liked movies
   - Gives +8 points per matching genre from liked movies

2. **Search History**
   - Automatically tracks your last 50 searches
   - Records movie IDs and languages from search results
   - Identifies frequently searched languages (2+ times)

3. **Language Preferences**
   - Learns your preferred languages from viewing patterns
   - Prioritizes movies in your top languages (+10 points)
   - Supports 20+ languages including regional Indian languages

#### Recommendation Scoring
Each recommended movie receives a score based on:
- **Liked Genre Match**: +8 points per matching genre (highest priority)
- **Language Match**: +10 points for top language, +8 for second, etc.
- **Genre Match**: +5 points per matching genre from history
- **Popularity**: +0.02 × popularity score
- **Rating**: +1.0 × vote average
- **Recent Release**: +3 points if released within 2 years

Movies are sorted by total score, and top 15-20 are displayed.

### Content-Based Filtering
For "Get Recommendations" feature, the engine analyzes:
- **Genres** - Movie categories
- **Keywords** - Thematic elements
- **Cast & Crew** - Actors and directors
- **Overview** - Plot descriptions

### Similarity Calculation
1. Extracts features from the selected movie
2. Compares with a database of popular movies
3. Uses **TF-IDF vectorization** to convert text to numerical features
4. Calculates **cosine similarity** between movies
5. Returns the top N most similar movies

### OpenCV Visual Analysis
- **Dominant Color Extraction** - K-means clustering on poster pixels
- **Brightness Analysis** - Average grayscale value
- **Complexity Score** - Edge detection to measure visual complexity

## 📁 Project Structure

```
Movie Recommendation Using Python/
├── movie_recommender.py              # Main application
├── requirements.txt                  # Python dependencies
├── .env.example                     # Environment variable template
├── .env                             # Your API key (create this)
├── .venv/                           # Virtual environment (created during setup)
├── README.md                        # This file
├── SEARCH_RECOMMENDATIONS_FEATURE.md # Feature documentation
├── movie_cache.pkl                  # Auto-generated cache file
├── search_history.pkl               # User search history (auto-generated)
└── liked_movies.pkl                 # User liked movies (auto-generated)
```

## 🎯 Features Breakdown

### Movie Information Displayed
- Title and poster
- Rating (out of 10) and vote count
- Release date
- Overview/synopsis
- Genres
- Popularity score
- Original language

### Browsing Modes
1. **🔍 Search Movies** - Search with personalized recommendations
2. **🌟 Browse Trending** - Weekly trending with filters
3. **⭐ Latest Movies** - Recently released with language filters
4. **🎭 Browse by Genre** - 19+ genres available
5. **🌍 Browse by Language** - 20+ languages supported
6. **📅 Browse by Year** - Any year from 1900 to present
7. **🎤 Dubbed Movies** - Find dubbed versions
8. **🎯 Advanced Filters** - Combine multiple filters

### Personalization System
- **Like/Unlike** movies across all modes
- **Search history** tracking (last 50 searches)
- **Language preference** learning
- **Genre preference** analysis from liked movies
- **Viewing history** for better recommendations
- **Liked movies collection** with sorting options:
  - View all liked movies
  - Group by browsing mode
  - Sort by recent or oldest first

### Recommendation Algorithms
1. **Personalized Recommendations**
   - Analyzes liked movies and search history
   - Multi-factor scoring system
   - Prioritizes user preferences
   - Returns top 15-20 personalized picks

2. **Content-Based Recommendations**
   - Analyzes up to 40 popular movies for comparison
   - Considers genres, keywords, cast, crew, and plot
   - Returns top 5-10 most similar movies
   - Real-time processing with caching for performance

### Data Privacy
- All data stored locally on your machine
- No external data sharing (except TMDb API calls)
- Clear history anytime with one click
- Persistent storage using pickle files

## 🔧 Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the same directory as `movie_recommender.py`
- Check that your API key is valid and active
- Make sure there are no extra spaces in the `.env` file

### Installation Issues
```bash
# If OpenCV fails to install, try:
pip install opencv-python-headless

# For M1/M2 Mac users:
pip install --upgrade pip
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Run on a different port:
streamlit run movie_recommender.py --server.port 8502
```

## 🌟 Recent Updates

### Version 2.0 - Personalization & Multi-Mode Browsing
- ✅ **Like System** - Like movies across all browsing modes
- ✅ **Personalized Recommendations** - Netflix-style recommendations based on liked movies
- ✅ **Search History Tracking** - Automatic tracking with language preference learning
- ✅ **Latest Movies Mode** - Browse recently released movies
- ✅ **Language Browsing** - 20+ languages including regional Indian languages
- ✅ **Year-Based Browsing** - Explore movies from any year
- ✅ **Dubbed Movies Mode** - Find dubbed versions of popular movies
- ✅ **Advanced Filters** - Combine genre, language, and year filters
- ✅ **Trending Filters** - Filter trending movies by language and year
- ✅ **Liked Movies Collection** - View and manage all your liked movies
- ✅ **Pagination** - Smooth navigation through large result sets

## 🚀 Future Enhancements

- [ ] User authentication and cloud sync
- [ ] Collaborative filtering based on user ratings
- [ ] Movie trailer integration
- [ ] Export recommendations and liked movies to PDF
- [ ] Social sharing features
- [ ] Watchlist with notifications for new releases
- [ ] Advanced analytics dashboard for viewing habits

## 📝 License

This project is open-source and available for educational purposes.

## 🙏 Credits

- Movie data provided by [The Movie Database (TMDb)](https://www.themoviedb.org/)
- Built with [Streamlit](https://streamlit.io/)
- Image processing with [OpenCV](https://opencv.org/)

## 📧 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify your TMDb API key is valid

---

**Enjoy discovering your next favorite movie! 🎬🍿**
