import pandas as pd
import numpy as np
from PIL import Image
from clip_retrieval.clip_client import ClipClient, Modality
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# 1️⃣ Initialize CLIP client
client_clip = ClipClient(
    url="http://localhost:6333",        # Your CLIP/Qdrant backend URL
    indice_name="jewellery_collection", # Name of your index/collection
    modality=Modality.IMAGE,            # Searching/embedding images
    num_images=50                        # Optional: number of results for queries
)

# 2️⃣ Initialize Qdrant client
client_qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "jewellery_skus"

# Optional: recreate collection (will overwrite if exists)
try:
    client_qdrant.recreate_collection(
        collection_name=collection_name,
        vector_size=512*2,  # CLIP image + text concatenated
        distance="Cosine"
    )
except Exception as e:
    print("Collection might already exist:", e)

# 3️⃣ Read CSV
df = pd.read_csv("sku_data.csv")

points = []

# 4️⃣ Loop through each row in CSV
for idx, row in df.iterrows():
    try:
        # Load image
        image_path = f"images/{row['image_filename']}"
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {row['image_filename']}: {e}")
        continue

    # Embed image using the new ClipClient query
    image_vector = client_clip.query(image=image_path)[0]['embedding']  # returns list of dicts

    # Embed text (combine design info)
    text_input = f"{row['main_design']} {row['design_type']} {row['line']} {row['cut']}"
    text_vector = client_clip.query(text=text_input)[0]['embedding']

    # Combine image + text vectors
    combined_vector = np.concatenate([image_vector, text_vector])

    # Prepare Qdrant point
    point = PointStruct(
        id=int(row['sku_id']),
        vector=combined_vector.tolist(),
        payload={
            "sku_code": row['sku_code'],
            "price": row['price'],
            "karat": row['karat'],
            "weight_g": row.get('weight_g', None),
            "size": row.get('size', None),
            "color": row['color'],
            "dimensions": row['dimensions']
        }
    )
    points.append(point)

# 5️⃣ Upsert all points to Qdrant
client_qdrant.upsert(collection_name=collection_name, points=points)

print("✅ Ingestion completed!")
