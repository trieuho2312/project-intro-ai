# 🎵 Music Recommendation App

A simple yet effective AI-powered web app that recommends songs with similar lyrics using TF-IDF and cosine similarity.

---

## ✅ Features

- Select a song and get 5 similar songs based on lyrics.
- Uses **TF-IDF vectorization** and **cosine similarity** for recommendation.
- Web interface built with **Streamlit**.
- Preprocessed from **Spotify Million Song Lyrics dataset**.

---

## 🧠 AI Technique Used

- **TF-IDF (Term Frequency - Inverse Document Frequency)**: Converts song lyrics into vector space.
- **Cosine Similarity**: Measures similarity between songs based on lyric vectors.

---

## 📁 Project Structure

```
music-recommendation-app/
├── app.py                  # Streamlit web interface
├── prepare_model.py        # Preprocessing & model building
├── requirements.txt
├── README.md

├── data/
│   └── spotify_millsongdata.csv

├── models/
│   ├── df_cleaned.pkl
│   ├── tfidf_matrix.pkl
│   └── cosine_sim.pkl

├── src/
│   ├── recommend.py
│   └── preprocess_utils.py
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess and train model
```bash
python prepare_model.py
```

### 3. Launch the web app
```bash
streamlit run app.py
```

---

## 🧩 Future Improvements

- Add filters (genre, artist, mood).
- Integrate user feedback (like/dislike) into future models.
- Replace TF-IDF with transformer-based models (e.g., BERT for lyrics).

---

## 📚 Dataset

Spotify Million Song Lyrics dataset  
(Only sampled 10,000 for efficiency)

---

## 👨‍💻 Author

Trieuho

