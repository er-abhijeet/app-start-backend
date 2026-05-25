# AI Photo Gallery - Backend System

This repository contains the backend infrastructure for the AI Photo Gallery system. It is a robust Python Flask application responsible for all the heavy lifting: facial recognition, image metadata extraction, AI-powered tagging, and maintaining the intelligent photo-sharing queues.

## 🧠 Core Features

- **Facial Recognition & Matching:** Uses `InsightFace` (ArcFace embeddings) to detect and recognize faces in uploaded photos.
- **Auto-Sharing & Stranger Conversion:** 
  - Automatically identifies registered users in photos and routes those photos to their personal shared queue.
  - Temporarily profiles "strangers". If a stranger later registers, all past photos containing their face are instantly linked to their new account!
- **AI Image Analysis:** Integrates with the **Google Gemini API** to analyze image content and generate intelligent search tags and descriptions.
- **Semantic Search:** Uses `SentenceTransformers` (all-MiniLM-L6-v2) to generate local text embeddings, allowing users to search their gallery using natural language.
- **Geofencing & Location Processing:** Calculates distances (Haversine formula) to manage geographic data and trigger location-based alerts.

## 🛠️ Tech Stack

- **Server:** Python, Flask, Waitress
- **Database:** MongoDB (PyMongo)
- **AI/ML:** InsightFace, SentenceTransformers, Google Generative AI (Gemini), OpenCV, SciPy
- **Image Processing:** Pillow (PIL)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- MongoDB instance (Atlas or local)
- Google Gemini API Key

### Installation

1. Clone this repository.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with the following variables:
   ```env
   MONGO_URI=your_mongodb_connection_string
   GEMINI_API_KEY=your_gemini_api_key
   ```
5. Run the server:
   ```bash
   python app.py
   ```

## 🧩 How it connects to the system
This is **2 of 3** repositories in the AI Photo Gallery ecosystem. 
- It serves as the central brain, receiving requests from the **Mobile App**.
- It provides data and geospatial coordinates to the **Web Portal** for map visualizations.
