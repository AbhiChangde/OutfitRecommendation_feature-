👗 AI-Powered Outfit Recommendation System

An end-to-end AI-driven fashion recommendation system that analyzes visual attributes from product images and generates context-aware outfit recommendations with natural language explanations.

🚀 Overview

Choosing outfits online is often overwhelming due to limited contextual understanding of style, color compatibility, and occasion relevance.
This project addresses that gap by combining computer vision, embeddings, and generative AI to recommend visually coherent outfits based on a given product image.

The system is designed as a real-world AI product prototype, focusing not only on model accuracy but also on user experience, explainability, and deployment.

🎯 Problem Statement

Fashion platforms show items in isolation, lacking outfit-level recommendations

Users struggle to visualize how products work together

Existing recommendations are rule-based or shallow (color-only, category-only)

💡 Solution

This application:

Extracts visual attributes from fashion images

Understands semantic similarity across products

Recommends complementary items (tops, bottoms, footwear, accessories)

Explains why an outfit works using natural language

🧠 Key Features

📸 Image-based fashion understanding

🔍 Semantic outfit matching using embeddings

🧾 Contextual recommendation explanations

🌐 Web-based interface for real-time interaction

☁️ Cloud deployment on AWS

🏗️ System Architecture

Input: Product image selected by the user

Visual Understanding:

Florence-2 extracts visual attributes

Semantic Retrieval:

MiniLM embeddings stored in ChromaDB

Similar and complementary items retrieved

Reasoning & Explanation:

Gemini 2.0 Flash generates contextual outfit suggestions

Delivery:

Flask backend serves recommendations to the UI

🧰 Tech Stack
AI & ML

Gemini 2.0 Flash – contextual reasoning & recommendation explanation

Florence-2 – visual feature extraction from images

MiniLM – semantic embeddings

Backend

Python

Flask

ChromaDB (vector database)

Cloud & Storage

AWS EC2 – application deployment

AWS S3 – fashion image storage (3,000+ catalog images)

Prototyping & Frontend

Claude Pro – rapid prototyping, UI flows, and front-end iteration

🧪 Deployment

The application is deployed on an AWS EC2 instance

Supports real-time inference and recommendations via a web interface

📌 Deployed on AWS EC2; link available on request

📦 Dataset

~3,000 fashion product images

Stored in AWS S3

Includes multiple categories (tops, bottoms, footwear, accessories)

🔍 Example Use Case

User uploads or selects a fashion item

System analyzes visual and semantic attributes

Complementary outfit pieces are retrieved

AI generates a recommendation explanation such as:

“This beige jacket pairs well with dark denim jeans due to contrast balance and casual styling consistency.”

📈 Learnings & Outcomes

Built an end-to-end AI product, not just a model

Learned how to:

Translate user problems into AI workflows

Balance accuracy with explainability

Deploy and serve AI systems in production-like environments

Gained hands-on experience in AI-assisted product prototyping using Claude Pro

🔮 Future Improvements

User preference learning & feedback loops

Occasion-based recommendations (formal, casual, festive)

Personalization using user history

Mobile-first UI

🙌 Acknowledgements

Open-source AI models and libraries

Inspiration from real-world fashion recommendation platforms


Rewrite this for a PM-heavy or engineering-heavy audience

Help you write a case-study version of this project

Just tell me 👍
