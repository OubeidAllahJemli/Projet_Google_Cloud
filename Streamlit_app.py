import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from google.cloud import bigquery

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .movie-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Cache functions for performance
@st.cache_resource
def load_bigquery_client():
    """Load BigQuery client"""
    return bigquery.Client(project="students-group2")

@st.cache_data
def load_model_artifacts():
    """Load the saved model artifacts"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        models_dir = script_dir / 'models' / 'item_similarity_2'
        
        with open(models_dir / 'item_similarity_with_confidence_weighting.pkl', 'rb') as f:
            item_similarity_df = pickle.load(f)
        
        df_movies = pd.read_pickle(models_dir / 'movies_metadata.pkl')
        
        with open(models_dir / 'movie_ids.pkl', 'rb') as f:
            movie_ids = pickle.load(f)
        
        return item_similarity_df, df_movies, movie_ids
    except FileNotFoundError:
        st.error("⚠️ Model artifacts not found. Please run the collaborative filtering notebook first.")
        return None, None, None

@st.cache_data
def load_ratings_data():
    """Load ratings data from BigQuery"""
    client = load_bigquery_client()
    query = """
    SELECT userId, movieId, rating
    FROM `master-ai-cloud.MoviePlatform.ratings`
    """
    df_ratings = client.query(query).to_dataframe()
    return df_ratings

def get_recommendations(user_ratings, item_similarity_df, df_movies, n_recommendations=10, min_common_users=5):
    """
    Generate movie recommendations based on user ratings
    
    Args:
        user_ratings: dict of {movieId: rating}
        item_similarity_df: DataFrame with item-item similarities
        df_movies: DataFrame with movie metadata
        n_recommendations: number of recommendations to return
        min_common_users: minimum number of common users for similarity
    
    Returns:
        DataFrame with recommended movies
    """
    predictions = {}
    
    for movie_id in item_similarity_df.index:
        if movie_id in user_ratings:
            continue
        
        weighted_sum = 0
        similarity_sum = 0
        
        for rated_movie_id, rating in user_ratings.items():
            if rated_movie_id in item_similarity_df.index:
                similarity = item_similarity_df.loc[movie_id, rated_movie_id]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            predictions[movie_id] = predicted_rating
    
    # Sort by predicted rating
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = sorted_predictions[:n_recommendations]
    
    # Create results dataframe
    results = []
    for movie_id, score in top_recommendations:
        movie_info = df_movies[df_movies['movieId'] == movie_id]
        if not movie_info.empty:
            results.append({
                'movieId': movie_id,
                'title': movie_info.iloc[0]['title'],
                'genres': movie_info.iloc[0]['genres'],
                'score': score
            })
    
    return pd.DataFrame(results)

def search_movies(query, df_movies):
    """Search for movies by title"""
    if not query:
        return pd.DataFrame()
    return df_movies[df_movies['title'].str.contains(query, case=False, na=False)]

# Main App
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Item-Based Collaborative Filtering avec Cosine Similarity")
    
    # Load data
    with st.spinner("Chargement des données..."):
        item_similarity_df, df_movies, movie_ids = load_model_artifacts()
        if item_similarity_df is None:
            st.stop()
        df_ratings = load_ratings_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Mode d'utilisation",
        ["🆕 Nouvel utilisateur", "👤 Utilisateur existant", "📊 Analyse du système"],
        index=0
    )
    
    # Main content based on mode
    if mode == "🆕 Nouvel utilisateur":
        show_new_user_mode(item_similarity_df, df_movies)
    elif mode == "👤 Utilisateur existant":
        show_existing_user_mode(item_similarity_df, df_movies, df_ratings)
    else:
        show_system_analysis(item_similarity_df, df_movies, df_ratings)

def show_new_user_mode(item_similarity_df, df_movies):
    """Interface for new users to rate movies and get recommendations"""
    st.header("🆕 Nouvel Utilisateur - Créez votre profil")
    
    st.markdown("""
    Notez quelques films pour obtenir des recommandations personnalisées. 
    Le système va **apprendre vos préférences** et affiner ses recommandations au fur et à mesure.
    """)
    
    # Initialize session state for user ratings
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Movie search
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Rechercher un film à noter", placeholder="Ex: Toy Story, Matrix...")
    
    # Display search results
    if search_query:
        search_results = search_movies(search_query, df_movies)
        if not search_results.empty:
            st.markdown("#### Résultats de recherche:")
            for idx, row in search_results.head(10).iterrows():
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                with col2:
                    rating = st.slider(
                        "Note",
                        0.5, 5.0, 3.0, 0.5,
                        key=f"rating_{row['movieId']}"
                    )
                with col3:
                    if st.button("Ajouter", key=f"add_{row['movieId']}"):
                        st.session_state.user_ratings[row['movieId']] = rating
                        st.success("✓")
                        st.rerun()
        else:
            st.info("Aucun film trouvé")
    
    # Display current ratings
    st.markdown("---")
    st.subheader(f"📝 Vos notes ({len(st.session_state.user_ratings)} films)")
    
    if st.session_state.user_ratings:
        for movie_id, rating in st.session_state.user_ratings.items():
            movie_info = df_movies[df_movies['movieId'] == movie_id]
            if not movie_info.empty:
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.write(f"**{movie_info.iloc[0]['title']}**")
                with col2:
                    st.write(f"⭐ {rating}/5.0")
                with col3:
                    if st.button("🗑️", key=f"remove_{movie_id}"):
                        del st.session_state.user_ratings[movie_id]
                        st.rerun()
    else:
        st.info("Vous n'avez pas encore noté de films. Utilisez la recherche ci-dessus!")
    
    # Generate recommendations
    st.markdown("---")
    n_recs = st.slider("Nombre de recommandations", 5, 20, 10)
    
    if st.button("🎯 Générer les recommandations", type="primary", disabled=len(st.session_state.user_ratings) == 0):
        if len(st.session_state.user_ratings) > 0:
            with st.spinner("Calcul des recommandations..."):
                recommendations = get_recommendations(
                    st.session_state.user_ratings,
                    item_similarity_df,
                    df_movies,
                    n_recommendations=n_recs
                )
                
                if not recommendations.empty:
                    st.success(f"✅ {len(recommendations)} recommandations générées!")
                    
                    # Display as cards
                    st.subheader("🎬 Films recommandés pour vous")
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{idx+1}. {row['title']}**")
                                st.caption(f"Genres: {row['genres']}")
                            with col2:
                                st.metric("Score", f"{row['score']:.2f}")
                else:
                    st.warning("Aucune recommandation trouvée. Notez plus de films!")
        else:
            st.warning("Veuillez noter au moins un film!")

def show_existing_user_mode(item_similarity_df, df_movies, df_ratings):
    """Interface for existing users from the database"""
    st.header("👤 Utilisateur Existant")
    
    # Get unique users
    unique_users = sorted(df_ratings['userId'].unique())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox(
            "Sélectionner un utilisateur",
            unique_users,
            index=0
        )
    with col2:
        n_recs = st.slider("Nombre de recommandations", 5, 20, 10)
    
    # Get user's ratings
    user_ratings_df = df_ratings[df_ratings['userId'] == selected_user]
    user_ratings_dict = dict(zip(user_ratings_df['movieId'], user_ratings_df['rating']))
    
    # Display user's profile
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Films notés", len(user_ratings_dict))
    with col2:
        avg_rating = user_ratings_df['rating'].mean()
        st.metric("Note moyenne", f"{avg_rating:.2f}")
    with col3:
        st.metric("Note max", f"{user_ratings_df['rating'].max():.1f}")
    
    # Show user's top rated movies
    st.subheader("🌟 Films les mieux notés par cet utilisateur")
    top_rated = user_ratings_df.nlargest(5, 'rating')
    top_rated_with_titles = top_rated.merge(df_movies, on='movieId', how='left')
    
    for idx, row in top_rated_with_titles.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{row['title']}**")
            st.caption(f"Genres: {row['genres']}")
        with col2:
            st.metric("Note", f"{row['rating']:.1f}/5.0")
    
    # Generate recommendations
    st.markdown("---")
    if st.button("🎯 Générer les recommandations", type="primary"):
        with st.spinner("Calcul des recommandations..."):
            recommendations = get_recommendations(
                user_ratings_dict,
                item_similarity_df,
                df_movies,
                n_recommendations=n_recs
            )
            
            st.success(f"✅ {len(recommendations)} recommandations générées!")
            
            st.subheader("🎬 Films recommandés")
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{idx+1}. {row['title']}**")
                        st.caption(f"Genres: {row['genres']}")
                    with col2:
                        st.metric("Score", f"{row['score']:.2f}")

def show_system_analysis(item_similarity_df, df_movies, df_ratings):
    """Show system analysis and statistics"""
    st.header("📊 Analyse du Système de Recommandation")
    
    # Overall statistics
    st.subheader("📈 Statistiques Globales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Utilisateurs", f"{df_ratings['userId'].nunique():,}")
    with col2:
        st.metric("Films", f"{len(df_movies):,}")
    with col3:
        st.metric("Ratings totaux", f"{len(df_ratings):,}")
    with col4:
        avg_rating = df_ratings['rating'].mean()
        st.metric("Note moyenne", f"{avg_rating:.2f}/5.0")
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Distribution des notes", "🎬 Films populaires", "🔗 Matrice de similarité"])
    
    with tab1:
        st.subheader("Distribution des notes")
        
        # Rating distribution
        fig = px.histogram(
            df_ratings,
            x='rating',
            nbins=20,
            title="Distribution des notes des utilisateurs",
            labels={'rating': 'Note', 'count': 'Fréquence'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Ratings per user
        ratings_per_user = df_ratings.groupby('userId').size().reset_index(name='count')
        fig2 = px.histogram(
            ratings_per_user,
            x='count',
            nbins=50,
            title="Distribution du nombre de notes par utilisateur",
            labels={'count': 'Nombre de films notés', 'count': 'Nombre d\'utilisateurs'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Films les plus populaires")
        
        # Most rated movies
        movie_counts = df_ratings.groupby('movieId').size().reset_index(name='num_ratings')
        movie_avg = df_ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
        movie_stats = movie_counts.merge(movie_avg, on='movieId')
        movie_stats = movie_stats.merge(df_movies[['movieId', 'title', 'genres']], on='movieId')
        movie_stats = movie_stats.sort_values('num_ratings', ascending=False)
        
        top_n = st.slider("Afficher top N films", 10, 50, 20)
        
        # Display top movies
        st.markdown("#### Films les plus notés")
        for idx, row in movie_stats.head(top_n).iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{row['title']}**")
                st.caption(f"{row['genres']}")
            with col2:
                st.metric("Notes", f"{row['num_ratings']:,}")
            with col3:
                st.metric("Moy.", f"{row['avg_rating']:.2f}")
        
        # Scatter plot
        fig3 = px.scatter(
            movie_stats.head(100),
            x='num_ratings',
            y='avg_rating',
            hover_data=['title'],
            title="Note moyenne vs Popularité (Top 100 films)",
            labels={'num_ratings': 'Nombre de notes', 'avg_rating': 'Note moyenne'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Matrice de similarité entre films")
        
        # Select a movie to see similar movies
        search_movie = st.text_input("Rechercher un film pour voir ses similarités", 
                                     placeholder="Ex: Toy Story")
        
        if search_movie:
            search_results = search_movies(search_movie, df_movies)
            if not search_results.empty:
                selected_movie = st.selectbox(
                    "Sélectionner un film",
                    search_results['title'].tolist()
                )
                
                # Get movie ID
                movie_id = search_results[search_results['title'] == selected_movie].iloc[0]['movieId']
                
                if movie_id in item_similarity_df.index:
                    # Get similarities
                    similarities = item_similarity_df.loc[movie_id].sort_values(ascending=False)
                    similarities = similarities[similarities.index != movie_id]  # Remove self
                    
                    # Top 10 similar movies
                    st.markdown(f"#### Films les plus similaires à **{selected_movie}**")
                    top_similar = similarities.head(10)
                    
                    similar_movies_data = []
                    for sim_movie_id, similarity in top_similar.items():
                        movie_info = df_movies[df_movies['movieId'] == sim_movie_id]
                        if not movie_info.empty:
                            similar_movies_data.append({
                                'Film': movie_info.iloc[0]['title'],
                                'Genres': movie_info.iloc[0]['genres'],
                                'Similarité': similarity
                            })
                    
                    df_similar = pd.DataFrame(similar_movies_data)
                    
                    # Display as table
                    st.dataframe(df_similar, use_container_width=True)
                    
                    # Visualization
                    fig4 = px.bar(
                        df_similar,
                        x='Similarité',
                        y='Film',
                        orientation='h',
                        title=f"Similarité avec {selected_movie}",
                        color='Similarité',
                        color_continuous_scale='Blues'
                    )
                    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.warning("Ce film n'est pas dans la matrice de similarité")

if __name__ == "__main__":
    main()