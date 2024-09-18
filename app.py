import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import read_pdf, split_document, topic_modeling, classify_and_cluster

# Create a word cloud for each cluster
def create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters):
    feature_names = vectorizer.get_feature_names_out()

    for cluster_num in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
        cluster_tfidf = tfidf_matrix[cluster_indices].toarray().sum(axis=0)

        # Create a dictionary for word frequencies
        word_freq = {feature_names[i]: cluster_tfidf[i] for i in range(len(feature_names)) if cluster_tfidf[i] > 0}

        # Generate the word cloud
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)

        # Use columns to arrange word clouds
        if cluster_num % 2 == 0:
            # Start a new row with two columns
            col1, col2 = st.columns(2)

        # Choose column to display in
        col = col1 if cluster_num % 2 == 0 else col2

        # Display the word cloud using matplotlib in the chosen column
        with col:
            st.subheader(f"Word Cloud for Cluster {cluster_num + 1}")
            fig, ax = plt.subplots(figsize=(5, 2.5))  # Adjust the size of the plot
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

# Set Streamlit page configuration to wide
st.set_page_config(layout="wide")

# Using HTML and CSS to style and center the title
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        width: 80%;
        margin: 5px auto; /* Center the div horizontally */
        padding: 20px 0;
    }
    </style>
    <div class="title">Document Clustering and Topic Modeling with Word Clouds</div>
    """,
    unsafe_allow_html=True
)

# File uploader for PDF
uploaded_files = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    all_doc_content_list = []

    # Read content from each PDF and split into sentences/paragraphs
    for uploaded_file in uploaded_files:
        doc_content = read_pdf(uploaded_file)
        doc_content_list = split_document(doc_content)
        all_doc_content_list.extend(doc_content_list)  # Combine all documents into one list

    if len(all_doc_content_list) > 1:
        # Slider to select number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=min(10, len(all_doc_content_list)), value=5)

        # Apply KMeans clustering and topic modeling
        tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(all_doc_content_list, n_clusters)

        st.header("Clusters and Topics")
        # Display cluster and topic information
        for i, cluster_topics in enumerate(topics):
            st.write(f"Cluster {i + 1}:")
            for topic in cluster_topics:
                st.write(topic)

        st.header("Cluster Word Clouds")
        # Create and display word clouds for each cluster
        create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
    else:
        st.write("Document is too small to cluster.")
