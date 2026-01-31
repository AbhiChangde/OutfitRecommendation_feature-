#!/usr/bin/env python
# coding: utf-8

# # 🛍️ Zara Fashion Recommendation - Complete Web Application
# ## Local Testing with Frontend + Backend
# 
# **What this does:**
# - Browse products with images in a beautiful web interface
# - Select any product and get AI-powered outfit recommendations  
# - See matching items with similarity scores
# - Ready for cloud deployment (VM + S3)
# 
# **How to use:**
# 1. Update your paths in Cell 2 below
# 2. Run all cells in order
# 3. Open http://localhost:5000 in your browser
# 4. Select a product to get recommendations!

# ## 📝 Configuration (Update Your Paths Here)

# In[1]:


import os

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Your Gemini API Key
os.environ['GEMINI_API_KEY'] = 'XXXX'

# ============================================================================
# EC2 CONFIGURATION - Images on S3
# ============================================================================

# S3 Configuration - IMAGES ARE ON S3
USE_S3 = True
IMAGE_BASE_URL = 'https://zara-product-images.s3.eu-north-1.amazonaws.com/zara/'  # CHANGE THIS!

# EC2 file paths (Linux paths, NOT Windows!)
CATALOG_CSV_PATH = '/home/ubuntu/florence2_descriptions.csv'
IMAGE_DIR = None  # Not used with S3
CHROMA_DB_PATH = '/home/ubuntu/chroma_fashion_db'

# Flask settings for EC2 - MUST be 0.0.0.0 to accept external connections
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 80

# ============================================================================

# Verify configuration
print("📋 Configuration:")
print(f"  API Key: {'✅ Set' if os.environ.get('GEMINI_API_KEY') else '❌ Missing'}")
print(f"  CSV: {'✅ Found' if os.path.exists(CATALOG_CSV_PATH) else '❌ Not found at ' + CATALOG_CSV_PATH}")

if USE_S3:
    print(f"  Images: ☁️  Using S3")
    print(f"  S3 URL: {IMAGE_BASE_URL}")
else:
    print(f"  Images: {'✅ Found' if IMAGE_DIR and os.path.exists(IMAGE_DIR) else '❌ Not found'}")
    if IMAGE_DIR and os.path.exists(IMAGE_DIR):
        img_count = len([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Total images: {img_count}")

print(f"  Mode: {'☁️  Cloud (S3)' if USE_S3 else '💻 Local files'}")

# ## 📦 Install Required Packages

# In[ ]:


# Note: Install packages manually before running:
# pip3 install --break-system-packages flask flask-cors chromadb google-generativeai sentence-transformers pandas pillow

# ## 📚 Import Libraries

# In[2]:


import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
import google.generativeai as genai

from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS

import threading
# IPython display not needed for script execution

print("✅ All libraries imported successfully")

# ## 🤖 Fashion Recommendation Agent (Backend Engine)

# In[3]:


# Cell: Updated FashionStylistAgent with local image support

import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Dict, Optional
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set your image directory
IMAGE_DIR = r"C:\Users\DELL\Documents\Zara Dataset\images\zara_for_colab"


class FashionStylistAgent:
    """AI Fashion Stylist with local image support"""
    
    def __init__(
        self,
        gemini_api_key: str,
        image_directory: str = IMAGE_DIR,
        chroma_persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the Fashion Stylist Agent"""
        self.gemini_api_key = gemini_api_key
        self.image_directory = image_directory
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_persist_directory)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        logger.info(f"Fashion Stylist Agent initialized")
        logger.info(f"Image directory: {self.image_directory}")
    
    def get_image_path(self, image_filename: str) -> Optional[str]:
        """
        Get full path for image filename
        
        Args:
            image_filename: Just the filename (e.g., 'jacket_001.jpg')
            
        Returns:
            Full path to image or None
        """
        if not image_filename or pd.isna(image_filename) or image_filename == '':
            return None
        
        full_path = os.path.join(self.image_directory, image_filename)
        
        if os.path.exists(full_path):
            return full_path
        else:
            logger.warning(f"Image not found: {full_path}")
            return None
    
    def load_catalog(self, csv_path: str) -> pd.DataFrame:
        """Load product catalog from CSV"""
        logger.info(f"Loading catalog from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Add product_id if not present
        if 'product_id' not in df.columns:
            df['product_id'] = df.index.astype(str)
        
        # Create full image paths
        if 'image_downloads' in df.columns:
            df['image_full_path'] = df['image_downloads'].apply(self.get_image_path)
            images_found = df['image_full_path'].notna().sum()
            logger.info(f"Found {images_found} images out of {len(df)} products")
        
        logger.info(f"Loaded {len(df)} products")
        return df
    
    def create_product_description(self, row: pd.Series) -> str:
        """Create comprehensive product description for embedding"""
        parts = []
        
        if 'name' in row and pd.notna(row['name']):
            parts.append(f"Name: {row['name']}")
        
        if 'description' in row and pd.notna(row['description']):
            parts.append(f"Description: {row['description']}")
        
        if 'florence2_description' in row and pd.notna(row['florence2_description']):
            parts.append(f"Visual: {row['florence2_description']}")
        
        if 'terms' in row and pd.notna(row['terms']):
            parts.append(f"Category: {row['terms']}")
        
        return " | ".join(parts)
    
    def ingest_catalog(self, df: pd.DataFrame, collection_name: str = "fashion_products"):
        """Ingest product catalog into ChromaDB"""
        logger.info("Starting catalog ingestion into ChromaDB")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Fashion product catalog"}
        )
        
        # Prepare data for ingestion
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            doc = self.create_product_description(row)
            documents.append(doc)
            
            metadata = {
                'product_id': str(row.get('product_id', idx)),
                'name': str(row.get('name', '')),
                'category': str(row.get('terms', '')),
            }
            
            # Store image filename (not full path)
            if 'image_downloads' in row and pd.notna(row['image_downloads']):
                metadata['image_filename'] = str(row['image_downloads'])
            
            metadatas.append(metadata)
            ids.append(f"product_{idx}")
        
        # Add to collection in batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            logger.info(f"Ingested batch {i//batch_size + 1}: {i}-{end_idx}")
        
        logger.info(f"Successfully ingested {len(documents)} products into ChromaDB")
    
    def load_image_for_gemini(self, image_filename: str) -> Optional[Image.Image]:
        """
        Load image from local directory
        
        Args:
            image_filename: Just the filename (e.g., 'jacket_001.jpg')
            
        Returns:
            PIL Image object or None
        """
        full_path = self.get_image_path(image_filename)
        
        if not full_path:
            return None
        
        try:
            img = Image.open(full_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            logger.info(f"Successfully loaded image: {image_filename}")
            return img
            
        except Exception as e:
            logger.warning(f"Could not load image {image_filename}: {e}")
            return None
    
    def get_outfit_recommendations(
        self,
        product_name: str,
        product_description: str,
        image_filename: Optional[str] = None,
        occasion: str = "casual everyday",
        style_vibe: str = "modern minimalist"
    ) -> Dict:
        """Get outfit recommendations from Gemini API"""
        logger.info(f"Getting outfit recommendations for: {product_name}")
        
        prompt = f"""You are an expert fashion stylist. Given a fashion item, suggest complementary pieces to create a complete, stylish outfit.

ANCHOR ITEM:
Name: {product_name}
Description: {product_description}

STYLING CONTEXT:
Occasion: {occasion}
Style Vibe: {style_vibe}

Please suggest 3-4 complementary items that would complete this outfit. For each item, provide:
1. Item category only including - tshirts, sweatshirts, trousers, jeans, shoes, jackets
2. Detailed description of style, color, material, and fit
3. Why it complements the anchor item

Format your response as a JSON array with this structure:
[
  {{
    "category": "trousers",
    "description": "slim-fit black cotton trousers with a tapered leg",
    "reasoning": "creates a sleek silhouette that balances the volume of the jacket"
  }},
  ...
]

Be specific about colors, materials, cuts, and styles."""
        
        try:
            # Try to load image if provided
            image_used = False
            content = [prompt]
            
            if image_filename:
                img = self.load_image_for_gemini(image_filename)
                if img:
                    content = [img, prompt]
                    image_used = True
                    logger.info("Image loaded successfully for Gemini analysis")
            
            # Generate content
            response = self.gemini_model.generate_content(content)
            response_text = response.text
            
            # Extract JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            recommendations = json.loads(json_str)
            
            logger.info(f"Received {len(recommendations)} recommendations from Gemini")
            return {
                'anchor_product': product_name,
                'occasion': occasion,
                'style_vibe': style_vibe,
                'recommendations': recommendations,
                'image_used': image_used
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise
    
    def search_similar_products(
        self,
        query: str,
        category_filter: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """Search for similar products using vector similarity"""
        logger.info(f"Vector search: '{query}'")
        
        where_filter = None
        if category_filter:
            where_filter = {"category": {"$eq": category_filter}}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        products = []
        for i in range(len(results['ids'][0])):
            product = {
                'product_id': results['metadatas'][0][i]['product_id'],
                'name': results['metadatas'][0][i]['name'],
                'category': results['metadatas'][0][i]['category'],
                'similarity_score': 1 - results['distances'][0][i],
                'image_filename': results['metadatas'][0][i].get('image_filename', ''),
                'image_path': self.get_image_path(results['metadatas'][0][i].get('image_filename', ''))
            }
            products.append(product)
        
        return products
    
    def complete_outfit(
        self,
        anchor_product_name: str,
        anchor_product_description: str,
        image_filename: Optional[str] = None,
        occasion: str = "casual everyday",
        style_vibe: str = "modern minimalist",
        top_k: int = 3
    ) -> Dict:
        """Complete workflow: Get recommendations and find matching products"""
        logger.info(f"Starting outfit completion for: {anchor_product_name}")
        
        # Step 1: Get recommendations from Gemini
        gemini_response = self.get_outfit_recommendations(
            anchor_product_name,
            anchor_product_description,
            image_filename,
            occasion,
            style_vibe
        )
        
        # Step 2: Search catalog for each recommendation
        outfit_items = []
        
        for rec in gemini_response['recommendations']:
            category = rec['category']
            description = rec['description']
            search_query = f"{category} {description}"
            
            matches = self.search_similar_products(
                query=search_query,
                category_filter=None,
                n_results=top_k
            )
            
            outfit_items.append({
                'recommendation': rec,
                'matched_products': matches
            })
        
        return {
            'anchor_product': {
                'name': anchor_product_name,
                'description': anchor_product_description,
                'image_filename': image_filename,
                'image_path': self.get_image_path(image_filename) if image_filename else None
            },
            'context': {
                'occasion': occasion,
                'style_vibe': style_vibe
            },
            'image_analysis_used': gemini_response.get('image_used', False),
            'outfit_items': outfit_items
        }
    
    def display_outfit(self, outfit: Dict):
        """Pretty print the complete outfit"""
        print("\n" + "="*80)
        print("OUTFIT COMPLETION")
        print("="*80)
        
        print(f"\nANCHOR ITEM: {outfit['anchor_product']['name']}")
        print(f"Description: {outfit['anchor_product']['description']}")
        
        if outfit['anchor_product'].get('image_filename'):
            print(f"Image: {outfit['anchor_product']['image_filename']}")
            if outfit['anchor_product'].get('image_path'):
                print(f"  Path: {outfit['anchor_product']['image_path']}")
        
        if outfit.get('image_analysis_used'):
            print("✅ Gemini analyzed product image")
        
        print(f"\nOCCASION: {outfit['context']['occasion']}")
        print(f"STYLE VIBE: {outfit['context']['style_vibe']}")
        
        print("\n" + "-"*80)
        print("RECOMMENDED OUTFIT ITEMS")
        print("-"*80)
        
        for idx, item in enumerate(outfit['outfit_items'], 1):
            rec = item['recommendation']
            print(f"\n{idx}. {rec['category'].upper()}")
            print(f"   AI Recommendation: {rec['description']}")
            print(f"   Reasoning: {rec['reasoning']}")
            
            print(f"\n   MATCHED PRODUCTS FROM CATALOG:")
            for match_idx, match in enumerate(item['matched_products'], 1):
                print(f"   {match_idx}) {match['name']}")
                print(f"      Similarity: {match['similarity_score']*100:.1f}%")
                if match.get('image_filename'):
                    print(f"      Image: {match['image_filename']}")
        
        print("\n" + "="*80)


print("✅ FashionStylistAgent class defined (with local image support)!")

# ## 🚀 Initialize the Recommendation Agent

# In[4]:


print("Initializing Fashion Recommendation Agent...")

agent = FashionStylistAgent(
    gemini_api_key=os.environ.get('GEMINI_API_KEY'),
    image_directory=IMAGE_DIR,
    chroma_persist_directory=CHROMA_DB_PATH
)

print("\n✅ Agent initialized and ready!")

# ## 📊 Load and Index Product Catalog

# In[5]:


# Load and ingest catalog
print("\nLoading catalog...")
catalog_df = agent.load_catalog(CATALOG_CSV_PATH)

print("\nIngesting into ChromaDB...")
agent.ingest_catalog(catalog_df)

print("\n✅ Catalog loaded and indexed!")
print(f"   Total products: {len(catalog_df)}")

# ## 🌐 Flask Web Application (Frontend + API)

# In[6]:


# Create Flask application
app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zara Fashion Recommendations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 40px;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { color: #7f8c8d; font-size: 1.1em; }
        .filter-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filter-group { display: flex; gap: 15px; flex-wrap: wrap; align-items: center; }
        .filter-group label { font-weight: 600; color: #2c3e50; }
        .filter-group select {
            padding: 8px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        }
        .product-card.selected {
            border: 3px solid #3498db;
            transform: scale(1.02);
        }
        .product-image {
            width: 100%;
            height: 300px;
            object-fit: cover;
            background: #f8f9fa;
        }
        .product-info { padding: 15px; }
        .product-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 1.1em;
        }
        .product-category {
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: capitalize;
        }
        .recommendations-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 40px;
            display: none;
        }
        .recommendations-section.active { display: block; }
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 1.2em;
            color: #7f8c8d;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .anchor-product {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .anchor-product img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
        }
        .category-section { margin-bottom: 40px; }
        .category-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
            text-transform: capitalize;
        }
        .recommendation-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .recommendation-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .similarity-score {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛍️ Zara Fashion Recommendations</h1>
            <p class="subtitle">Select a product to get personalized outfit recommendations</p>
        </header>
        
        <div class="filter-section">
            <div class="filter-group">
                <label for="occasion">Occasion:</label>
                <select id="occasion">
                    <option value="casual everyday">Casual</option>
                    <option value="formal business">Formal</option>
                    <option value="party night out">Party</option>
                    <option value="business professional">Business</option>
                    <option value="weekend relaxed">Weekend</option>
                </select>
                
                <label for="style">Style:</label>
                <select id="style">
                    <option value="modern minimalist">Modern</option>
                    <option value="classic timeless">Classic</option>
                    <option value="urban streetwear">Streetwear</option>
                    <option value="elegant sophisticated">Elegant</option>
                    <option value="minimalist clean">Minimalist</option>
                </select>
                
                <button onclick="loadProducts()">🔄 Refresh Products</button>
            </div>
        </div>
        
        <div id="products-container">
            <h2 style="color: #2c3e50; margin-bottom: 20px;">Available Products</h2>
            <div class="products-grid" id="products-grid"></div>
        </div>
        
        <div id="recommendations-section" class="recommendations-section">
            <h2 style="color: #2c3e50; margin-bottom: 20px;">Your Personalized Outfit</h2>
            <div id="recommendations-content"></div>
        </div>
    </div>
    
    <script>
        let selectedProduct = null;
        let allProducts = [];
        
        window.onload = function() { loadProducts(); };
        
        async function loadProducts() {
            const grid = document.getElementById('products-grid');
            grid.innerHTML = '<div class="loading"><div class="spinner"></div>Loading products...</div>';
            
            try {
                const response = await fetch('/api/products');
                allProducts = await response.json();
                
                grid.innerHTML = '';
                allProducts.forEach(product => {
                    const card = createProductCard(product);
                    grid.appendChild(card);
                });
            } catch (error) {
                grid.innerHTML = '<div class="loading">Error loading products.</div>';
                console.error('Error:', error);
            }
        }
        
        function createProductCard(product) {
            const card = document.createElement('div');
            card.className = 'product-card';
            card.onclick = () => selectProduct(product, card);
            
            card.innerHTML = `
                <img src="/images/${product.image_filename}" 
                     alt="${product.name}"
                     class="product-image"
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'200\' height=\'200\'%3E%3Crect fill=\'%23f0f0f0\' width=\'200\' height=\'200\'/%3E%3Ctext fill=\'%23999\' x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\'%3ENo Image%3C/text%3E%3C/svg%3E'">
                <div class="product-info">
                    <div class="product-name">${product.name}</div>
                    <div class="product-category">${product.category}</div>
                </div>
            `;
            return card;
        }
        
        async function selectProduct(product, cardElement) {
            document.querySelectorAll('.product-card').forEach(card => {
                card.classList.remove('selected');
            });
            cardElement.classList.add('selected');
            
            selectedProduct = product;
            
            const recsSection = document.getElementById('recommendations-section');
            const recsContent = document.getElementById('recommendations-content');
            recsSection.classList.add('active');
            recsContent.innerHTML = '<div class="loading"><div class="spinner"></div>Generating recommendations...</div>';
            
            recsSection.scrollIntoView({ behavior: 'smooth' });
            
            try {
                const occasion = document.getElementById('occasion').value;
                const style = document.getElementById('style').value;
                
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        product: product,
                        occasion: occasion,
                        style_vibe: style
                    })
                });
                
                const outfit = await response.json();
                displayRecommendations(outfit);
            } catch (error) {
                recsContent.innerHTML = '<div class="loading">Error generating recommendations.</div>';
                console.error('Error:', error);
            }
        }
        
        function displayRecommendations(outfit) {
            const content = document.getElementById('recommendations-content');
            
            let html = `
                <div class="anchor-product">
                    <img src="/images/${outfit.anchor_product.image_filename}" alt="${outfit.anchor_product.name}">
                    <div>
                        <h3>Your Selection</h3>
                        <p><strong>${outfit.anchor_product.name}</strong></p>
                        <p style="color: #7f8c8d; margin-top: 5px;">${outfit.anchor_product.description || ''}</p>
                        <p style="margin-top: 10px;">
                            <span style="background: #e8f4f8; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">
                                📅 ${outfit.context.occasion}
                            </span>
                            <span style="background: #f8e8f8; padding: 5px 10px; border-radius: 5px;">
                                ✨ ${outfit.context.style_vibe}
                            </span>
                        </p>
                    </div>
                </div>
            `;
            
            outfit.outfit_items.forEach(item => {
                const rec = item.recommendation;
                html += `
                    <div class="category-section">
                        <h3 class="category-title">${rec.category}</h3>
                        <p style="color: #555; margin-bottom: 15px; font-style: italic;">${rec.description}</p>
                        <p style="color: #777; margin-bottom: 20px; font-size: 0.95em;">💡 ${rec.reasoning}</p>
                        <div class="products-grid">
                `;
                
                item.matched_products.forEach(match => {
                    const score = (match.similarity_score * 100).toFixed(0);
                    html += `
                        <div class="recommendation-card">
                            <img src="/images/${match.image_filename}" alt="${match.name}">
                            <div class="product-name">${match.name}</div>
                            <div class="product-category" style="margin-top: 5px;">${match.category}</div>
                            <span class="similarity-score">Match: ${score}%</span>
                        </div>
                    `;
                });
                
                html += `</div></div>`;
            });
            
            content.innerHTML = html;
        }
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/images/<path:filename>')
def serve_image(filename):
    if USE_S3:
        from flask import redirect
        return redirect(f"{IMAGE_BASE_URL}{filename}")
    else:
        return send_from_directory(IMAGE_DIR, filename)

# Global variable to store catalog for quick access
_catalog_df = None

@app.route('/api/products')
def get_products():
    """Get random products for display"""
    global _catalog_df
    try:
        # Load catalog if not already loaded
        if _catalog_df is None:
            _catalog_df = agent.load_catalog(CATALOG_CSV_PATH)
        
        # Get random sample
        sample = _catalog_df.sample(n=min(20, len(_catalog_df)))
        products = []
        
        for _, row in sample.iterrows():
            products.append({
                'product_id': str(row.get('product_id', '')),
                'name': str(row.get('name', '')),
                'description': str(row.get('description', '')),
                'category': str(row.get('terms', '')),
                'image_filename': str(row.get('image_downloads', ''))
            })
        
        return jsonify(products)
    except Exception as e:
        print(f"Error in get_products: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get outfit recommendations"""
    try:
        data = request.json
        product = data.get('product')
        occasion = data.get('occasion', 'casual everyday')
        style_vibe = data.get('style_vibe', 'modern minimalist')
        
        # Get recommendations using the agent
        outfit = agent.complete_outfit(
            anchor_product_name=product['name'],
            anchor_product_description=product.get('description', ''),
            image_filename=product.get('image_filename'),
            occasion=occasion,
            style_vibe=style_vibe,
            top_k=3
        )
        
        return jsonify(outfit)
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

print("✅ Flask app configured")
print(f"📱 Ready to start at http://localhost:{FLASK_PORT}")

# ## ▶️ Run the Web Application
# 
# **Click "Run" on this cell to start the server, then open http://localhost:5000 in your browser!**
# 
# To stop the server: **Kernel → Interrupt**

# In[ ]:


# ============================================================================
# MAIN EXECUTION - Run Flask Server
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING FASHION RECOMMENDATION APP ON EC2")
    print("="*70)
    print(f"\n📱 The app will be available at:")
    print(f"   http://YOUR_EC2_PUBLIC_IP:{FLASK_PORT}")
    print(f"\n⚙️  Features:")
    print(f"   • Browse products with images from S3")
    print(f"   • Select occasion and style preferences")
    print(f"   • Get AI-powered outfit recommendations")
    print(f"\n🛑 To stop: Press Ctrl+C")
    print("="*70 + "\n")
    
    # Run Flask server
    # Note: debug=False for production, set to True only for debugging
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)

# End of script

# ## ☁️ Cloud Deployment Guide (VM + S3)
# 
# ### Step 1: Upload Images to S3
# ```python
# import boto3
# 
# s3 = boto3.client('s3',
#     aws_access_key_id='YOUR_KEY',
#     aws_secret_access_key='YOUR_SECRET'
# )
# 
# bucket = 'your-bucket'
# for img in Path(IMAGE_DIR).glob('*.jpg'):
#     s3.upload_file(
#         str(img),
#         bucket,
#         f'images/{img.name}',
#         ExtraArgs={'ACL': 'public-read'}
#     )
# ```
# 
# ### Step 2: Update Configuration (Cell 2)
# ```python
# USE_S3 = True
# IMAGE_BASE_URL = 'https://your-bucket.s3.amazonaws.com/images/'
# FLASK_HOST = '0.0.0.0'
# ```
# 
# ### Step 3: Deploy to VM
# ```bash
# # Install dependencies
# pip3 install flask flask-cors chromadb google-generativeai pandas
# 
# # Run notebook or convert to script
# # Open firewall: sudo ufw allow 5000
# ```
