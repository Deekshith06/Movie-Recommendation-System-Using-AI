# 🎬 NextWatch

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![ML](https://img.shields.io/badge/ML-Cosine_Similarity-green?style=flat-square&logo=scikit-learn)
![Data](https://img.shields.io/badge/Movies-5000+-orange?style=flat-square)

Content-based movie recommendation system that suggests films based on **plot, genre, and cast similarity**. Built with Python, Streamlit, and the OMDB API for real-time metadata.

---

## 🔄 How It Works

```mermaid
graph TD
    %% Styling - High Contrast (Dark Text on Light Background)
    classDef offline fill:#ffffff,stroke:#333333,stroke-width:2px,color:#000000;
    classDef runtime fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000000;
    classDef storage fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000000;
    classDef external fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000000;

    subgraph Offline ["Offline Training (One-Time Process)"]
        A[TMDB 5000 Dataset] -->|Clean & Tokenize| B(Text Tags)
        B -->|CountVectorizer| C{Vector Space}
        C -->|Cosine Similarity| D[Similarity Matrix]
    end

    subgraph Storage ["Model Storage"]
        D -->|Serialize| E[("similarities.pkl")]
        A -.->|Extract Metadata| F[("moviess.pkl")]
    end

    subgraph Runtime ["Runtime Application (Streamlit)"]
        User([User Selects Movie]) -->|Input| Engine
        
        subgraph Engine ["Recommendation Engine"]
            E -.->|Load Matrix| Logic[Find Nearest Vectors]
            F -.->|Load Metadata| Logic
            Logic -->|Top 5 Candidate IDs| Filter[Filter & Sort]
        end

        Filter -->|Fetch Posters| API[OMDB API]
        API -->|Return Images| UI[Movie Cards UI]
    end

    %% Key Connections
    class A,B,C,D offline;
    class User,Engine,Logic,Filter,UI runtime;
    class E,F storage;
    class API external;
```

### 🧠 Understanding the Process
Think of this as a **"Matchmaker" for movies**:
1.  **The "DNA"**: We mash up keywords, genres, and cast into a "soup" of text.
2.  **The Math**: We turn that soup into a mathematical **Vector** (a list of numbers).
3.  **The Match**: When you pick a movie, we find other vectors pointing in the **same direction** (Cosine Similarity).
---

## � Quick Start

```bash
git clone https://github.com/Deekshith06/NextWatch.git
cd NextWatch
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Note**: Add your OMDB API key to `.streamlit/secrets.toml` to fetch movie posters.

---

## 📂 Project Structure

```
├── app.py                  # Main Application logic
├── requirements.txt        # Dependencies
├── .streamlit/             # API Configuration
│   └── secrets.toml
└── data/                   # ML Models & Data
    ├── moviess.pkl
    └── similarities.pkl
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit (Custom CSS) |
| ML Model | Scikit-learn (Cosine Similarity) |
| Data Processing | Pandas, NumPy |
| API | OMDB (Real-time Metadata) |

---

## 📊 Features

| Feature | Description |
|--------|-------|
| **Smart Search** | Instantly find any movie in the 5000+ database. |
| **Hybrid Ranking** | Combines content similarity with weighted ratings for quality suggestions. |
| **Live Metadata** | Fetches posters, plots, and cast details on the fly. |
| **Responsive UI** | Mobile-friendly grid layout with dark mode aesthetics. |

---

## 👤 Author

**Seelaboyina Deekshith**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Deekshith06)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deekshith030206)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:seelaboyinadeekshith@gmail.com)

---

> ⭐ Star this repo if it helped you!
