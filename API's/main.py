
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import base64
import pandas as pd
import snowflake.connector
import requests
from typing import List, Optional
from datetime import datetime
import numpy as np
from fastapi import FastAPI
import google.generativeai as genai
from pathlib import Path
import tomllib  # Python 3.11+

# Adjust path to point inside the 'api' folder
secrets_path = Path(__file__).parent / "api" / "secrets.toml"

with secrets_path.open("rb") as f:
    secrets = tomllib.load(f)

# Configure Gemini API
genai.configure(api_key=secrets["genai"]["api_key"])

app = FastAPI()

# Access secrets
plant_id_api_key = secrets["plant_id"]["api_key"]
snowflake_config = secrets["snowflake"]

@app.get("/")
def read_root():
    return {"message": "API is running"}

# --- Request Models ---
class QARequest(BaseModel):
    question: str
    lat: float
    lon: float

class LocationRequest(BaseModel):
    lat: float
    lon: float
    radius_km: Optional[float] = 5.0
    month: Optional[int] = None  # if none, use current month

class QuestRequest(BaseModel):
    lat: float
    lon: float
    date: Optional[str] = None  # YYYY-MM-DD, default today
    difficulty: Optional[str] = "easy"  # or medium, hard
    features: Optional[List[str]] = None  # e.g. ["opposite leaves", "purple flowers"]

def connect_snowflake():
    return snowflake.connector.connect(**snowflake_config)

def fetch_local_plants(lat: float, lon: float, radius_km: float, month: Optional[int]):
    conn = connect_snowflake()

    lat_min = lat - (radius_km / 111)
    lat_max = lat + (radius_km / 111)
    lon_min = lon - (radius_km / (111 * abs(np.cos(np.radians(lat)))))
    lon_max = lon + (radius_km / (111 * abs(np.cos(np.radians(lat)))))

    if not month:
        month = datetime.now().month

    query = f""" 
    SELECT
    obs.SCIENTIFICNAME,
    obs.LOCALITY,
    obs.EVENTDATE,
    charac.FLOWERCOLOR,
    charac.FRUITCOLOR,
    charac.GROWTHHABIT,
    charac.LIFESPAN,
    charac.DURATION,
    charac.FAMILY
FROM ARQ.SOURCE.GBIF_OBSERVATION obs
JOIN ARQ.SOURCE.PLANT_CHARACTERISTICS charac
    ON UPPER(obs.VERBATIMSCIENTIFICNAME) = UPPER(charac.SCIENTIFICNAME)
WHERE obs.DECIMALLATITUDE BETWEEN 40.66775495495496 AND 40.757845045045045
  AND obs.DECIMALLONGITUDE BETWEEN -74.06542707164776 AND -73.94657292835224
  AND EXTRACT(MONTH FROM obs.EVENTDATE) = 8
"""



    df = pd.read_sql(query, conn)
    conn.close()

    return df

def summarize_plants_for_prompt(df: pd.DataFrame) -> str:
    """
    Creates a concise summary string for LLM prompt from plant data.
    """
    if df.empty:
        return "No local plant data available."

    summaries = []
    for _, row in df.iterrows():
        parts = [
            f"Species: {row['SCIENTIFICNAME']}",
            f"Observed at: {row['LOCALITY'] or 'unknown location'}",
            f"Event date: {str(row['EVENTDATE'])[:10] if row['EVENTDATE'] else 'unknown'}",
            f"Flower color: {row['FLOWERCOLOR'] or 'unknown'}",
            f"Fruit color: {row['FRUITCOLOR'] or 'unknown'}",
            f"Growth habit: {row['GROWTHHABIT'] or 'unknown'}",
            f"Lifespan: {row['LIFESPAN'] or 'unknown'}",
            f"Duration: {row['DURATION'] or 'unknown'}",
            f"Family: {row['FAMILY'] or 'unknown'}",
            f"Leaf arrangement: {row.get('LEAFARRANGEMENT', 'unknown')}",
            f"Leaf margin: {row.get('LEAFMARGIN', 'unknown')}"
        ]
        summaries.append("; ".join(parts))
    return "\n".join(summaries[:10])  # limit to first 10 for prompt brevity


# --- Endpoints ---

@app.post("/llm-question")
def llm_question(data: QARequest):
    df = fetch_local_plants(data.lat, data.lon, radius_km=5.0, month=None)

    kg_summary = summarize_plants_for_prompt(df)

    prompt = f"""
You are a helpful expert botanist assistant.

Here is a summary of plants observed near latitude {data.lat} and longitude {data.lon}:

{kg_summary}

User Question: "{data.question}"

Based on the above, provide a concise, accurate answer about plant types, bloom periods, colors, leaf and flower features.
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)

    if response.candidates and response.candidates[0].content.parts:
        return {"answer": response.text.strip()}
    else:
        return {"answer": "The model did not return any answer. Possibly blocked or empty response."}


@app.post("/identify-plant")
async def identify_plant(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "images": [image_b64],
        "organs": ["leaf", "flower"],
        "similar_images": True
    }
    headers = {
        "Content-Type": "application/json",
        "Api-Key": plant_id_api_key
    }

    response = requests.post("https://api.plant.id/v2/identify", json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()

    # Extract features for user guidance if available
    features = {}
    if "suggestions" in data and len(data["suggestions"]) > 0:
        best_suggestion = data["suggestions"][0]
        # Some plant.id responses include detailed leaf/flower features under "plant_details"
        details = best_suggestion.get("plant_details", {})
        leaf_info = details.get("leaf", {})
        flower_info = details.get("flower", {})

        features = {
            "leaf_arrangement": leaf_info.get("leaf_arrangement"),
            "leaf_shape": leaf_info.get("leaf_shape"),
            "leaf_margin": leaf_info.get("leaf_margin"),
            "flower_color": flower_info.get("color"),
            "flower_shape": flower_info.get("shape"),
            "flower_symmetry": flower_info.get("symmetry")
        }

    return {
        "result": data,
        "key_features_to_note": features
    }


@app.post("/nearby-plants")
def nearby_plants(data: LocationRequest):
    df = fetch_local_plants(data.lat, data.lon, radius_km=data.radius_km, month=data.month)
    if df.empty:
        return {"plants": [], "message": "No plants found nearby for specified parameters."}

    plants_list = []
    for _, row in df.iterrows():
        plants_list.append({
            "species": row["SCIENTIFICNAME"],
            "location": row["LOCALITY"],
            "event_date": str(row["EVENTDATE"])[:10] if row["EVENTDATE"] else None,
            "flower_color": row["FLOWERCOLOR"],
            "fruit_color": row["FRUITCOLOR"],
            "growth_habit": row["GROWTHHABIT"],
            "lifespan": row["LIFESPAN"],
            "duration": row["DURATION"],
            "family": row["FAMILY"],
            "leaf_arrangement": row.get("LEAFARRANGEMENT"),
            "leaf_margin": row.get("LEAFMARGIN"),
        })

    return {"plants": plants_list, "count": len(plants_list)}


@app.post("/generate-quests")
def generate_quests(data: QuestRequest):
    # Use date or default to today
    date_str = data.date or datetime.now().strftime("%Y-%m-%d")
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month

    df = fetch_local_plants(data.lat, data.lon, radius_km=5.0, month=month)
    if df.empty:
        return {"quests": [], "message": "No plant data available for this location and date."}

    # Simple quest generation based on requested features & difficulty
    quests = []

    # If features requested, filter species by feature keywords
    filtered_df = df
    if data.features:
        mask = pd.Series([True] * len(df))
        for feat in data.features:
            feat_lower = feat.lower()
            mask &= df.apply(lambda row: any(feat_lower in str(val).lower() for val in row if val), axis=1)
        filtered_df = df[mask]

    # Build quests
    for _, row in filtered_df.head(10).iterrows():
        quest_desc = f"Find a plant species '{row['SCIENTIFICNAME']}' "
        # Add some feature hints
        hints = []
        if row.get("LEAFARRANGEMENT"):
            hints.append(f"with leaf arrangement: {row['LEAFARRANGEMENT']}")
        if row.get("FLOWERCOLOR"):
            hints.append(f"flower color: {row['FLOWERCOLOR']}")
        if hints:
            quest_desc += "that has " + ", ".join(hints)
        quests.append(quest_desc)

    return {
        "date": date_str,
        "location": {"lat": data.lat, "lon": data.lon},
        "difficulty": data.difficulty,
        "quests": quests
    }
