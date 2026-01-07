from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model artifacts
item_similarity_df = None
df_movies = None
model_metadata = {}

def load_model_artifacts():
    """Load all model artifacts on startup"""
    global item_similarity_df, df_movies, model_metadata
    
    print("Loading model artifacts...")
    
    # Load item similarity matrix
    with open('models/item_similarity.pkl', 'rb') as f:
        item_similarity_df = pickle.load(f)
    print(f"✓ Loaded item similarity matrix: {item_similarity_df.shape}")
    
    # Load movies metadata
    df_movies = pd.read_pickle('models/movies_metadata.pkl')
    print(f"✓ Loaded movies metadata: {len(df_movies)} movies")
    
    # Load metadata
    try:
        with open('models/metadata.pkl', 'rb') as f:
            model_metadata = pickle.load(f)
        print(f"✓ Loaded model metadata")
    except:
        model_metadata = {
            'model_type': 'Item-Based Collaborative Filtering',
            'created_at': datetime.now().isoformat()
        }
    
    print("Model loaded successfully!")

def get_recommendations(user_ratings, n_recommendations=10):
    """
    Get movie recommendations based on user's ratings
    
    Parameters:
    - user_ratings: dict of {movieId: rating}
    - n_recommendations: number of recommendations to return
    
    Returns:
    - list of recommended movies with scores
    """
    scores = {}
    similarity_sums = {}
    
    for movie_id, rating in user_ratings.items():
        movie_id = int(movie_id)
        
        if movie_id not in item_similarity_df.index:
            continue
            
        # Get similar movies
        similar_movies = item_similarity_df[movie_id]
        
        for other_movie_id, similarity in similar_movies.items():
            # Skip if already rated
            if other_movie_id in [int(mid) for mid in user_ratings.keys()]:
                continue
            
            # Weighted score
            if other_movie_id not in scores:
                scores[other_movie_id] = 0
                similarity_sums[other_movie_id] = 0
            
            scores[other_movie_id] += similarity * rating
            similarity_sums[other_movie_id] += similarity
    
    # Normalize scores
    recommendations = {}
    confidences = {}
    for movie_id in scores:
        if similarity_sums[movie_id] > 0:
            recommendations[movie_id] = scores[movie_id] / similarity_sums[movie_id]
            confidences[movie_id] = similarity_sums[movie_id]
    
    # Sort by confidence-weighted score
    sorted_recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1] * confidences[x[0]],
        reverse=True
    )[:n_recommendations]
    
    # Get movie details
    recommended_movie_ids = [movie_id for movie_id, _ in sorted_recommendations]
    recommended_movies = df_movies[df_movies['movieId'].isin(recommended_movie_ids)].copy()
    
    # Add scores
    score_dict = dict(sorted_recommendations)
    recommended_movies['score'] = recommended_movies['movieId'].map(score_dict)
    recommended_movies = recommended_movies.sort_values('score', ascending=False)
    
    # Convert to list of dicts
    result = []
    for _, row in recommended_movies.iterrows():
        result.append({
            'movieId': int(row['movieId']),
            'title': row['title'],
            'genres': row['genres'],
            'score': float(row['score'])
        })
    
    return result

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Item-Based Collaborative Filtering',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Health check',
            '/recommend': 'POST - Get recommendations',
            '/movies': 'GET - List all movies',
            '/movie/<id>': 'GET - Get movie details',
            '/similar/<id>': 'GET - Get similar movies'
        }
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Get movie recommendations
    
    Request body:
    {
        "user_ratings": {
            "1": 5.0,
            "50": 4.5,
            "260": 4.0
        },
        "n_recommendations": 10
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if 'user_ratings' not in data:
            return jsonify({
                'error': 'Missing user_ratings in request body'
            }), 400
        
        user_ratings = data['user_ratings']
        n_recommendations = data.get('n_recommendations', 10)
        
        # Validate n_recommendations
        if n_recommendations < 1 or n_recommendations > 50:
            return jsonify({
                'error': 'n_recommendations must be between 1 and 50'
            }), 400
        
        # Convert string keys to integers
        user_ratings = {int(k): float(v) for k, v in user_ratings.items()}
        
        # Get recommendations
        recommendations = get_recommendations(user_ratings, n_recommendations)
        
        return jsonify({
            'success': True,
            'user_ratings_count': len(user_ratings),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/movies', methods=['GET'])
def list_movies():
    """List all available movies with optional search"""
    try:
        # Get query parameters
        search = request.args.get('search', '').lower()
        genre = request.args.get('genre', '').lower()
        limit = int(request.args.get('limit', 100))
        
        # Filter movies
        movies = df_movies.copy()
        
        if search:
            movies = movies[movies['title'].str.lower().str.contains(search)]
        
        if genre:
            movies = movies[movies['genres'].str.lower().str.contains(genre)]
        
        # Limit results
        movies = movies.head(limit)
        
        # Convert to list
        result = []
        for _, row in movies.iterrows():
            result.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres']
            })
        
        return jsonify({
            'success': True,
            'count': len(result),
            'movies': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get details for a specific movie"""
    try:
        movie = df_movies[df_movies['movieId'] == movie_id]
        
        if movie.empty:
            return jsonify({
                'error': f'Movie ID {movie_id} not found'
            }), 404
        
        movie_data = movie.iloc[0]
        
        return jsonify({
            'success': True,
            'movie': {
                'movieId': int(movie_data['movieId']),
                'title': movie_data['title'],
                'genres': movie_data['genres']
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/similar/<int:movie_id>', methods=['GET'])
def get_similar_movies(movie_id):
    """Get movies similar to a given movie"""
    try:
        n_similar = int(request.args.get('n', 10))
        
        if movie_id not in item_similarity_df.index:
            return jsonify({
                'error': f'Movie ID {movie_id} not found'
            }), 404
        
        # Get similar movies
        similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:n_similar+1]
        
        # Get movie details
        result = []
        for mid, similarity in similar_movies.items():
            movie = df_movies[df_movies['movieId'] == mid]
            if not movie.empty:
                movie_data = movie.iloc[0]
                result.append({
                    'movieId': int(mid),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity': float(similarity)
                })
        
        return jsonify({
            'success': True,
            'movie_id': movie_id,
            'similar_movies': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': item_similarity_df is not None,
        'n_movies': len(df_movies) if df_movies is not None else 0,
        'similarity_matrix_shape': item_similarity_df.shape if item_similarity_df is not None else None,
        'metadata': model_metadata,
        'timestamp': datetime.now().isoformat()
    })

# Load model on startup
print("Starting Movie Recommendation API...")
load_model_artifacts()
print("API ready!")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
