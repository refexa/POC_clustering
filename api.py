from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.decomposition import PCA
from fastapi.middleware.cors import CORSMiddleware
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import io
import numpy as np
from fastapi.responses import JSONResponse
from typing import List
from utils import read_pdf, split_document, classify_and_cluster
import base64
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, or specify domains in a list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# localhost//cluster-documents/
@app.get("/")
async def root():
    return {"message": "We are up and running"}

@app.post("/cluster-documents/")
async def cluster_documents(files: List[UploadFile] = File(...), n_clusters: int = 5):
    """
    API endpoint to upload PDF files, cluster their content, and perform topic modeling.

    Args:
        files (List[UploadFile]): A list of uploaded PDF files.
        n_clusters (int): The number of clusters for KMeans.

    Returns:
        JSONResponse: Clusters and topics extracted from the PDF content, along with word cloud images in base64 format.
    """

    all_doc_content_list = []

    # Read and process content from each PDF
    for file in files:
        try:
            contents = await file.read()
            doc_content = read_pdf(io.BytesIO(contents))
            doc_content_list = split_document(doc_content)
            all_doc_content_list.extend(doc_content_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {file.filename}")

    if len(all_doc_content_list) < 2:
        raise HTTPException(status_code=400, detail="Insufficient content to cluster.")

    # Apply KMeans clustering and topic modeling
    tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(all_doc_content_list, n_clusters)

    # Prepare cluster information and topics for the response
    cluster_data = []
    wordclouds = {}
    feature_names = vectorizer.get_feature_names_out()

    for i, cluster_topics in enumerate(topics):
        cluster_data.append({
            "cluster": i + 1,
            "topics": [str(topic) for topic in cluster_topics]
        })

        # Generate word cloud for each cluster
        cluster_indices = [idx for idx, label in enumerate(clusters) if label == i]
        cluster_tfidf = tfidf_matrix[cluster_indices].toarray().sum(axis=0)

        # Create word frequency dictionary
        word_freq = {feature_names[j]: cluster_tfidf[j] for j in range(len(feature_names)) if cluster_tfidf[j] > 0}

        # Generate word cloud image
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)

        # Save word cloud image to a buffer
        buf = io.BytesIO()
        plt.figure(figsize=(5, 2.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the word cloud image to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        wordclouds[f"cluster_{i + 1}_wordcloud"] = image_base64

    # Return clusters, topics, and word clouds
    return JSONResponse(content={"clusters": cluster_data, "wordclouds": wordclouds})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



