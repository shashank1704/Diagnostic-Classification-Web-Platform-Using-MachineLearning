from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample texts
text1 = "I love programming in Python."
text2 = "How are you1?"

# Preprocess and vectorize the text using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the texts
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Print the cosine similarity
print("Cosine Similarity:", cosine_sim[0][0])

'''import numpy as np
from numpy.linalg import norm

A = np.array(( 2 , 1 , 2 ))
B = np.array(( 3 , 4 , 2 ))

cosine = np.dot(A,B) / (norm(A) * norm(B))  #Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)

print ( "Cosine Similarity:" , cosine)'''