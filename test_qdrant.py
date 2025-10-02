from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

# 1️⃣ Connect to Qdrant running in Docker
client = QdrantClient(host="localhost", port=6333)

# 2️⃣ Create a collection (only first time)
client.recreate_collection(
    collection_name="jewellery_collection",
    vectors_config=VectorParams(size=512, distance="Cosine")  # size = embedding dim
)

# 3️⃣ Insert sample embeddings + metadata
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, ...],  # replace with real CLIP embedding
        payload={"sku_id": "123", "sku_code": "RNG001", "price": 5000}
    )
]
client.upsert(collection_name="jewellery_collection", points=points)

# 4️⃣ Query a vector
results = client.search(
    collection_name="jewellery_collection",
    query_vector=points[0].vector,
    limit=5
)

for res in results:
    print(f"ID: {res.id}, Score: {res.score}, Payload: {res.payload}")
