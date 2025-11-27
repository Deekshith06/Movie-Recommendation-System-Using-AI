# 🎬 Movie Recommendation System Using AI

A modern, intelligent movie recommendation system built with **Streamlit**, **OpenCV**, and **The Movie Database (TMDb) API**. Get personalized movie recommendations with advanced content-based filtering and visual poster analysis.

## ✨ Features

- 🔍 **Live Movie Search** with pagination
- 🌟 **Trending & Latest Movies** with filters
- 🎭 **Browse by Genre, Language, Year**
- 🎤 **Dubbed Movies** finder
- 🎯 **Advanced Multi-Filter** combinations
- ❤️ **Like System** for personalized recommendations
- 🎬 **Netflix-style Recommendations** based on your preferences
- 📜 **Smart History Tracking** (last 50 searches)
- 🎨 **Visual Analysis** with OpenCV
- ⚡ **Fast & Responsive** with intelligent caching

## 🛠️ Technologies Used

Streamlit • OpenCV • TMDb API • scikit-learn • Pandas • NumPy

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
nano .env
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

### Search & Recommendations
- Search movies by name or leave empty for personalized picks
- Like movies to build your preference profile
- Get instant recommendations based on your taste

### Browse Modes
1. **Search Movies** - With personalized suggestions
2. **Trending** - Weekly popular with filters
3. **Latest** - Recent releases
4. **Genre** - 19+ categories
5. **Language** - 20+ languages
6. **Year** - 1900 to present
7. **Dubbed** - Find dubbed versions
8. **Advanced Filters** - Combine multiple criteria

## 🧠 How It Works

**Personalized System**: Tracks liked movies and search history to learn your preferences. Scores recommendations based on genre matches (+8), language preference (+10), popularity, ratings, and recency.

**Content-Based Filtering**: Uses TF-IDF and cosine similarity to analyze genres, keywords, cast, crew, and plot for finding similar movies.

**Visual Analysis**: OpenCV extracts dominant colors, brightness, and complexity from movie posters.

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

## 🔧 Troubleshooting

**API Key Issues**: Ensure `.env` file is in the same directory with valid API key and no extra spaces.

**Installation Issues**:
```bash
# If OpenCV fails:
pip install opencv-python-headless

# For M1/M2 Mac:
pip install --upgrade pip
pip install -r requirements.txt
```

**Port Already in Use**:
```bash
streamlit run movie_recommender.py --server.port 8502
```

## 🙏 Credits

Movie data by [TMDb](https://www.themoviedb.org/) • Built with [Streamlit](https://streamlit.io/) • Powered by [OpenCV](https://opencv.org/)

## 🤝 Contributing

Issues and pull requests welcome.

## 📫 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deekshith030206)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Deekshith06)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:seelaboyinadeekshith@gmail.com)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/deekshith06)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/deekshith06)

---

⭐ From [Deekshith06](https://github.com/Deekshith06) | Building the future with AI 🚀

**Enjoy discovering your next favorite movie! 🎬🍿**
