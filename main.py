import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import AgglomerativeClustering
from utils import read_pdf, split_document, create_wordcloud


# Set Streamlit page configuration to wide
st.set_page_config(layout="wide")


# Function to classify and cluster based on the selected algorithm
def classify_and_cluster(all_doc_content_list, n_clusters=None, cluster_algo="K-Means"):
    # Convert documents into TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_doc_content_list)

    if cluster_algo == "K-Means":
        model = KMeans(n_clusters=n_clusters)
    elif cluster_algo == "DBSCAN":
        model = DBSCAN()
    elif cluster_algo == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)

    # Fit the model and predict clusters
    clusters = model.fit_predict(tfidf_matrix.toarray())

    # Perform topic modeling or extract topics based on frequent terms
    topics = [vectorizer.inverse_transform(tfidf_matrix[i]) for i in range(len(clusters))]

    return tfidf_matrix, clusters, topics, vectorizer


# Function to perform clustering and display results
def perform_clustering(cluster_algo, all_doc_content_list, n_clusters):
    start_time = time()  # Measure the time taken
    tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(all_doc_content_list, n_clusters, cluster_algo)
    end_time = time()

    st.header("Clusters and Topics")
    for i, cluster_topics in enumerate(topics):
        st.write(f"Cluster {i + 1}:")
        topics_str = ", ".join([str(topic) for topic in cluster_topics])
        st.write(topics_str)

    st.header("Cluster Word Clouds")
    create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)

    # Show time taken
    st.write(f"Clustering took {end_time - start_time:.2f} seconds.")


# Navigation function to switch between pages
def show_nav():
    pages = ["Home", "Do Clustering", "Compare Clustering Algorithms", "Quiz", "Feedback"]
    choice = st.sidebar.radio("Navigate", pages)
    return choice


# Page 1: Home
def show_home():
    st.markdown(
        """
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            width: 80%;
            margin: 5px auto;
            padding: 20px 0;
        }
        </style>
        <div class="title">Document Clustering and Topic Modeling with Word Clouds</div>
        """,
        unsafe_allow_html=True
    )
    st.write("Welcome to the Document Clustering and Topic Modeling application!")


# Page 2: Do Clustering
def show_clustering():
    st.title("Do Clustering")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) > 0:
        all_doc_content_list = []

        for uploaded_file in uploaded_files:
            doc_content = read_pdf(uploaded_file)
            doc_content_list = split_document(doc_content)
            all_doc_content_list.extend(doc_content_list)

        if len(all_doc_content_list) > 1:
            # Form validation and cluster algorithm selection
            cluster_algo = st.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN", "Agglomerative Clustering"])

            if cluster_algo != "DBSCAN":
                n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)
            else:
                n_clusters = None

            if st.button("Run Clustering"):
                perform_clustering(cluster_algo, all_doc_content_list, n_clusters)
        else:
            st.write("Document is too small to cluster.")


# Page 3: Compare Clustering Algorithms
def compare_clustering_algorithms():
    st.title("Compare Clustering Algorithms")

    st.write("This page will showcase the comparison between different clustering algorithms in terms of performance metrics and visualizations. It will also include analytics like Silhouette Score, cluster sizes, etc.")

    # Placeholder for performance comparison (you can expand this with metrics, visualizations, etc.)
    st.write("Coming soon!")


# Page 4: Quiz
def show_quiz():
    st.title("Clustering Quiz")
    st.write("This page will contain quizzes based on clustering knowledge. Test your understanding with multiple-choice questions!")
    # Add your quiz implementation here (questions, scoring, etc.)
    st.write("Quiz coming soon!")


# Page 5: Feedback
def show_feedback():
    st.title("Feedback")
    st.write("We would love to hear your feedback!")
    feedback = st.text_area("Enter your feedback here")
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")


# Main App Controller
def main():
    page = show_nav()

    if page == "Home":
        show_home()
    elif page == "Do Clustering":
        show_clustering()
    elif page == "Compare Clustering Algorithms":
        compare_clustering_algorithms()
    elif page == "Quiz":
        show_quiz()
    elif page == "Feedback":
        show_feedback()


if __name__ == "__main__":
    main()
