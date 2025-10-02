import pandas as pd
import numpy as np
from PIL import Image
import torch
import clip
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# 1️⃣ Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"✅ CLIP model loaded on: {device}")

# 2️⃣ Initialize Qdrant client
client_qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "jewellery_skus_test"

# 3️⃣ Read your CSV file
df = pd.read_csv("sku_data.csv")
print("📊 CSV columns:", df.columns.tolist())
print(f"📋 Total SKUs in CSV: {len(df)}")

# 4️⃣ Clean column names (remove spaces)
df.columns = df.columns.str.strip()
print("🔧 Cleaned columns:", df.columns.tolist())

# 5️⃣ Check if images folder exists
images_folder = "images"
if not os.path.exists(images_folder):
    print(f"❌ Images folder '{images_folder}' not found!")
    exit()

# 6️⃣ List all files in images folder
print(f"\n📁 Files in '{images_folder}' folder:")
image_files = os.listdir(images_folder)
for img_file in image_files:
    print(f"   - {img_file}")

# 7️⃣ SPECIFIC TEST: Check only these 2 SKUs
test_skus = ["2BOXCHANKYA1", "BALL2"]
print(f"\n🔍 Testing specific SKUs: {test_skus}")

points = []

for sku_name in test_skus:
    print(f"\n--- Processing {sku_name} ---")
    
    # Find the SKU in CSV
    sku_row = df[df['image_filename'] == sku_name]
    
    if sku_row.empty:
        print(f"❌ SKU '{sku_name}' not found in CSV")
        continue
    
    row = sku_row.iloc[0]
    print(f"✅ Found in CSV: SKU {row['sku_id']} - {row['sku_code']}")
    
    # Check image file with different extensions
    image_found = False
    image_path = None
    
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        test_path = f"{images_folder}/{sku_name}{ext}"
        if os.path.exists(test_path):
            image_path = test_path
            image_found = True
            print(f"✅ Image found: {test_path}")
            break
    
    if not image_found:
        print(f"❌ No image found for {sku_name} with common extensions")
        continue
    
    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        print(f"✅ Image loaded and preprocessed: {image.size}")
        
        # Generate CLIP embedding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_vector = image_features.cpu().numpy().astype(np.float32).flatten()
        
        print(f"✅ CLIP embedding generated: {len(image_vector)} dimensions")
        
        # Prepare text description
        text_description = f"{row.get('main_design', '')} {row.get('design_type', '')} {row.get('line', '')} {row.get('cut', '')}".strip()
        print(f"📝 Text description: '{text_description}'")
        
        # Create Qdrant point
        point = PointStruct(
            id=int(row['sku_id']),
            vector=image_vector.tolist(),
            payload={
                "sku_id": int(row['sku_id']),
                "sku_code": row['sku_code'],
                "price": row['price'],
                "karat": row['karat'],
                "color": row['color'],
                "design_type": row.get('design_type', ''),
                "cut": row.get('cut', ''),
                "image_filename": sku_name,
                "text_description": text_description
            }
        )
        points.append(point)
        print(f"✅ Point prepared for Qdrant")
        
    except Exception as e:
        print(f"❌ Error processing {sku_name}: {e}")
        continue

# 8️⃣ Create collection and insert points
if points:
    try:
        client_qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{collection_name}' created")
        
        client_qdrant.upsert(collection_name=collection_name, points=points)
        print(f"✅ Successfully inserted {len(points)} test SKUs into Qdrant")
        
        # Verify insertion
        collection_info = client_qdrant.get_collection(collection_name=collection_name)
        print(f"📊 Collection info: {collection_info.points_count} points")
        
    except Exception as e:
        print(f"❌ Qdrant error: {e}")
else:
    print("❌ No points were processed successfully")

print(f"\n🎯 TEST SUMMARY:")
print(f"   - Target SKUs: {test_skus}")
print(f"   - Successfully processed: {len(points)}")
print(f"   - Collection: {collection_name}")