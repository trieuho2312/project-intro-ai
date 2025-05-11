import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Check if CSV exists
csv_path = 'spotify_millsongdata.csv'
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found!")
    exit(1)

print(f"Found CSV file: {csv_path}")

# Load and process data
try:
    print("Loading CSV data...")
    df = pd.read_csv(csv_path).sample(10000)
    print(f"Loaded {len(df)} rows")

    # Drop link column
    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

    # Text cleaning - simplified without NLTK
    print("Cleaning text (simplified)...")

    def simple_clean_text(text):
        # Convert to string, lowercase, and remove non-alphabetic chars
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        words = text.split()
        words = [word for word in words if len(word) > 2]
        return " ".join(words)

    df['cleaned_text'] = df['text'].apply(simple_clean_text)
    print("Text cleaning completed")

    # Vectorization
    print("Creating TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Cosine similarity
    print("Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

    # Save files
    print("Saving all required files...")

    # Save paths
    pickle_path = os.path.abspath('models/df_cleaned.pkl')
    tfidf_path = os.path.abspath('models/tfidf_matrix.pkl')
    sim_path = os.path.abspath('models/cosine_sim.pkl')

    os.makedirs('models', exist_ok=True)

    # Save with joblib
    print(f"Saving DataFrame to: {pickle_path}")
    joblib.dump(df, pickle_path)

    print(f"Saving TF-IDF matrix to: {tfidf_path}")
    joblib.dump(tfidf_matrix, tfidf_path)

    print(f"Saving similarity matrix to: {sim_path}")
    joblib.dump(cosine_sim, sim_path)

    # Verify files were created
    print("\nVerifying files were created:")
    for file_path in [pickle_path, tfidf_path, sim_path]:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)} created ({os.path.getsize(file_path) / 1024 / 1024:.2f} MB)")
        else:
            print(f"✗ {os.path.basename(file_path)} NOT created!")

    print("\nAll preprocessing completed successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
