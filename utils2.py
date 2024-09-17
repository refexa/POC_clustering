import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

from PyPDF2 import PdfReader
import pandas as pd


# Helper function to read PDF file
def read_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to split document content into meaningful sections
def split_document(doc_content):
    # Split by multiple newlines or multiple spaces (for paragraphs)
    paragraphs = re.split(r'\n{2,}|\s{2,}', doc_content)

    # Tokenize each paragraph into sentences
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    # Filter out empty sentences or very short sentences
    sentences = [sentence for sentence in sentences if len(sentence.strip()) > 2]

    return sentences


# Topic modeling using LDA
def topic_modeling(tfidf_matrix, feature_names, n_topics=5):
    lda = LDA(n_components=n_topics, random_state=0)
    lda.fit(tfidf_matrix)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {idx + 1}: {' '.join(topic_words)}")
    return topics


# # Function to classify and cluster document content
# def classify_and_cluster(doc_content_list, n_clusters=5):
#     # Preprocess document content using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
#     tfidf_matrix = vectorizer.fit_transform(doc_content_list)

#     # KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
#     clusters = kmeans.fit_predict(tfidf_matrix)

#     # Topic Modeling for each cluster
#     topics = []
#     for cluster_num in range(n_clusters):
#         cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
#         cluster_tfidf = tfidf_matrix[cluster_indices]
#         topics.append(topic_modeling(cluster_tfidf, vectorizer.get_feature_names_out()))

#     return tfidf_matrix, clusters, topics


# Function to classify and cluster document
def classify_and_cluster(doc_content_list, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(doc_content_list)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Get topics for each cluster using topic modeling
    topics = []
    for cluster_num in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]

        # If a cluster has data points, generate topics for it
        if len(cluster_indices) > 0:
            cluster_tfidf = tfidf_matrix[cluster_indices]
            cluster_topics = topic_modeling(cluster_tfidf, vectorizer.get_feature_names_out())
            topics.append(cluster_topics)
        else:
            topics.append([])  # Empty list for clusters with no data points

    return tfidf_matrix, clusters, topics, vectorizer
