import streamlit as st
import pandas as pd
import requests
import base64
import google.generativeai as genai           # LLM/KG Arborquest App
import json
import snowflake.connector

import networkx as nx

from typing import List
from infomap import Infomap
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm
from typing import Dict
from datetime import datetime


# ---------------- SETUP ----------------
# Load secrets
sf = st.secrets["snowflake"]
genai.configure(api_key=st.secrets["gemini"]["api_key"])
plant_id_api_key = st.secrets["plantid"]["api_key"]


# Snowflake connections
conn = snowflake.connector.connect(
   user=sf["user"],
   password=sf["password"],
   account=sf["account"],
   warehouse=sf["warehouse"],
   database=sf["database"],
   schema=sf["schema"],
   role=sf["role"]
)
# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Plant Intelligence Explorer", layout="wide")
st.title("🌿 Plant Intelligence Explorer")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Choose App Mode")
mode = st.sidebar.radio(
    "Mode", 
    ["Home", "LLM Plant Q&A", "Plant.id Image ID", "ArborQuest Quest Mode", "Algorithms", "Seasonal Plant Predictions"]
)

lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 47.60, format="%.5f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, -122.33, format="%.5f")

# Show radius slider only for quest modes
if mode in ["ArborQuest Quest Mode", "Custom Quest Builder"]:
    radius_km = st.sidebar.slider("Search Radius (km)", 1, 100, 30)
else:
    radius_km = None
# ----- Constants -----
DEFAULT_SNOWFLAKE_TABLE = "ARQ.PUBLIC.OBSERVATIONSWITHCHARACTERISTICS"
DEFAULT_PLANT_ID_COLUMN = "scientificname"
DEFAULT_CHARACTERISTIC_COLUMNS = [
    "flowercolor", "foliagecolor", "fruitcolor", "bloomperiod",
    "growthhabit", "droughttolerance", "fireresistance", "nativestatus",
    "order", "familie", "genus", "species", "category",
]

# Snowflake connection details (replace or better use env vars in prod)
SNOWFLAKE_CONFIG = {
    "user": "DAVID_VALENTA",
    "password": "Snowflakearborquest123",
    "account": "ndsoebe-rai-dev-ecosystem-aws-us-east-2-consumer",
    "warehouse": "arq_wh",
    "database": "ARQ",
    "schema": "PUBLIC",
    "authenticator": "username_password_mfa",
}


# ----- Functions -----


def get_connection():
    """Establish Snowflake connection."""
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


    
def build_filter_where_clause(genus=None, growth_habit=None, bloom_period=None):
    conditions = []
    if genus:
        genus_list = ",".join(f"'{g}'" for g in genus)
        conditions.append(f"GENUS IN ({genus_list})")
    if growth_habit:
        habit_list = ",".join(f"'{g}'" for g in growth_habit)
        conditions.append(f"GROWTHHABIT IN ({habit_list})")
    if bloom_period:
        period_list = " OR ".join(f"BLOOMPERIOD ILIKE '%{p}%'" for p in bloom_period)
        conditions.append(f"({period_list})")
    return " AND ".join(conditions)

def get_plant_graph_from_snowflake(genus=None, growth_habit=None, bloom_period=None):
    """Build filtered plant genus graph from Snowflake."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        base_query = f"""
            SELECT SCIENTIFICNAME, GENUS
            FROM {DEFAULT_SNOWFLAKE_TABLE}
            WHERE SCIENTIFICNAME IS NOT NULL AND GENUS IS NOT NULL
        """
        where_clause = build_filter_where_clause(genus, growth_habit, bloom_period)
        if where_clause:
            base_query += f" AND {where_clause}"

        cursor.execute(base_query)
        records = cursor.fetchall()

        genus_to_plants = {}
        for scientific_name, genus in records:
            genus_to_plants.setdefault(genus, []).append(scientific_name)

        G = nx.Graph()
        all_plants = [sci for sci, _ in records]
        G.add_nodes_from(all_plants)

        for plants in genus_to_plants.values():
            for i in range(len(plants)):
                for j in range(i + 1, len(plants)):
                    G.add_edge(plants[i], plants[j])
        return G
    finally:
        cursor.close()
        conn.close()



def create_or_replace_table(conn, table_name, schema_sql):
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE OR REPLACE TABLE {table_name} ({schema_sql})")
    finally:
        cursor.close()


def insert_many(conn, table_name, columns, data):
    cursor = conn.cursor()
    try:
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
        cursor.executemany(sql, data)
    finally:
        cursor.close()


def calculate_jaccard_similarity(G: nx.Graph, threshold=0.01):
    st.write("Calculating Jaccard similarity...")
    jaccard_scores = []
    nodes = list(G.nodes())

    for u, v in tqdm(combinations(nodes, 2), total=len(nodes)*(len(nodes)-1)//2):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        union = neighbors_u | neighbors_v
        if not union:
            continue
        intersection = neighbors_u & neighbors_v
        score = len(intersection) / len(union)
        if score > threshold:
            jaccard_scores.append((u, v, score))
    return jaccard_scores


def calculate_cosine_similarity(G: nx.Graph):
    st.write("Calculating Cosine similarity...")
    nodes = list(G.nodes())
    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
    sim_matrix = cosine_similarity(adj_matrix)

    cosine_scores = []
    n = len(nodes)
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score > 0:
                cosine_scores.append((nodes[i], nodes[j], score))
    return cosine_scores


def create_similarity_tables(conn):
    create_or_replace_table(conn, "PLANT_JACCARD_SIMILARITIES",
                            "PLANT_1 VARCHAR, PLANT_2 VARCHAR, JACCARD_SIMILARITY FLOAT")
    create_or_replace_table(conn, "PLANT_COSINE_SIMILARITIES",
                            "PLANT_1 VARCHAR, PLANT_2 VARCHAR, COSINE_SIMILARITY FLOAT")


def run_similarity_computations(genus=None, growth_habit=None, bloom_period=None):
    G = get_plant_graph_from_snowflake(genus, growth_habit, bloom_period)
    st.write(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    conn = get_connection()
    try:
        create_similarity_tables(conn)
        jaccard_data = calculate_jaccard_similarity(G, threshold=0.01)
        insert_many(conn, "PLANT_JACCARD_SIMILARITIES", ["PLANT_1", "PLANT_2", "JACCARD_SIMILARITY"], jaccard_data)

        cosine_data = calculate_cosine_similarity(G)
        insert_many(conn, "PLANT_COSINE_SIMILARITIES", ["PLANT_1", "PLANT_2", "COSINE_SIMILARITY"], cosine_data)
    finally:
        conn.close()



def create_community_table(conn, table_name):
    create_or_replace_table(conn, table_name,
                            "SCIENTIFICNAME VARCHAR, COMMUNITY_ID INTEGER")


def insert_communities(conn, table_name, partition: Dict[str, int]):
    data = [(plant, community) for plant, community in partition.items()]
    insert_many(conn, table_name, ["SCIENTIFICNAME", "COMMUNITY_ID"], data)


def run_infomap_communities(genus=None, growth_habit=None, bloom_period=None):
    G = get_plant_graph_from_snowflake(genus, growth_habit, bloom_period)
    st.write(f"Running Infomap on graph with {G.number_of_nodes()} nodes...")
    im = Infomap()
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    id_to_node = {i: node for node, i in node_to_id.items()}
    for u, v in G.edges():
        im.add_link(node_to_id[u], node_to_id[v])
    im.run()

    partition = {id_to_node[node.node_id]: node.module_id for node in im.nodes}

    conn = get_connection()
    try:
        create_community_table(conn, "PLANT_INFOMAP_COMMUNITIES")
        insert_communities(conn, "PLANT_INFOMAP_COMMUNITIES", partition)
    finally:
        conn.close()


def run_louvain_communities(genus=None, growth_habit=None, bloom_period=None):
    G = get_plant_graph_from_snowflake(genus, growth_habit, bloom_period)
    st.write(f"Running Louvain on graph with {G.number_of_nodes()} nodes...")
    partition = community_louvain.best_partition(G)
    conn = get_connection()
    try:
        create_community_table(conn, "PLANT_LOUVAIN_COMMUNITIES")
        insert_communities(conn, "PLANT_LOUVAIN_COMMUNITIES", partition)
    finally:
        conn.close()


def search_similarities(plant_name: str, table_name: str, similarity_col: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = f"""
            SELECT PLANT_1, PLANT_2, {similarity_col}
            FROM {table_name}
            WHERE PLANT_1 = %s OR PLANT_2 = %s
            ORDER BY {similarity_col} DESC
            LIMIT 10
        """
        cursor.execute(query, (plant_name, plant_name))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()


def search_communities(plant_name: str, community_table: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Get community for plant
        cursor.execute(f"""
            SELECT COMMUNITY_ID FROM {community_table} WHERE SCIENTIFICNAME = %s
        """, (plant_name,))
        community = cursor.fetchone()
        if not community:
            return None, []

        community_id = community[0]

        # Get all plants in that community
        cursor.execute(f"""
            SELECT SCIENTIFICNAME FROM {community_table} WHERE COMMUNITY_ID = %s
        """, (community_id,))
        plants = [row[0] for row in cursor.fetchall()]
        return community_id, plants
    finally:
        cursor.close()
        conn.close()

def load_full_table(table_name: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=colnames)




# ----- Streamlit UI -----
if mode == "Algorithms":
    st.set_page_config(page_title="🌿 Plant Knowledge Graph Quest", layout="wide")
    st.title("🌿 Plant Knowledge Graph Quest")

    with st.sidebar:
        st.header("Filter Plants Before Running Algorithms")

        # Example filter options — update or load dynamically as needed
        genus_options = ["Quercus", "Pinus", "Acer", "Betula"]
        growth_habit_options = ["Tree", "Shrub", "Herbaceous"]
        bloom_period_options = ["Spring", "Summer", "Fall", "Winter"]

        genus_filter = st.multiselect("Select Genus (optional):", genus_options)
        growth_habit_filter = st.multiselect("Select Growth Habit (optional):", growth_habit_options)
        bloom_period_filter = st.multiselect("Select Bloom Period (optional):", bloom_period_options)

        st.header("Build Knowledge Graph")
        if st.button("Build & Deploy Graph Model"):
            st.info("🧱 Graph build placeholder – no action taken.")

        st.header("Run Similarity & Community Computations")
        if st.button("Run Infomap Communities"):
            with st.spinner("Running Infomap and storing results..."):
                run_infomap_communities(
                    genus=genus_filter,
                    growth_habit=growth_habit_filter,
                    bloom_period=bloom_period_filter
                )
            st.success("Infomap communities computed and stored.")

        if st.button("Run Louvain Communities"):
            with st.spinner("Running Louvain and storing results..."):
                run_louvain_communities(
                    genus=genus_filter,
                    growth_habit=growth_habit_filter,
                    bloom_period=bloom_period_filter
                )
            st.success("Louvain communities computed and stored.")

        if st.button("Run Similarity Computations"):
            with st.spinner("Calculating similarities and storing results..."):
                run_similarity_computations(
                    genus=genus_filter,
                    growth_habit=growth_habit_filter,
                    bloom_period=bloom_period_filter
                )
            st.success("Similarity scores computed and stored.")

    st.markdown("---")
    st.info("Use the sidebar buttons to run similarity/community calculations, and then search for plants by name.")

    st.subheader("📊 Explore Community or Similarity Tables")

    table_options = [
        "PLANT_INFOMAP_COMMUNITIES",
        # "PLANT_LOUVAIN_COMMUNITIES",  # Removed as requested
        "PLANT_JACCARD_SIMILARITIES",
        "PLANT_COSINE_SIMILARITIES",
    ]

    selected_table = st.selectbox("Select a table to view:", table_options)
    search_query = st.text_input("Search within selected table (optional):").lower().strip()

    if selected_table:
        try:
            df_table = load_full_table(selected_table)
            if not df_table.empty:
                if search_query:
                    df_table = df_table[df_table.apply(
                        lambda row: row.astype(str).str.lower().str.contains(search_query).any(), axis=1
                    )]
                st.dataframe(df_table, use_container_width=True)
            else:
                st.warning("The selected table is empty.")
        except Exception as e:
            st.error(f"Failed to load table: {e}")













# ----- Full Table Viewer -----

# ---- PAGE LAYOUT ---- #
if mode == "Seasonal Plant Predictions":
    st.set_page_config(page_title="🌿 Seasonal Plant Predictions", layout="wide")
    st.title("🌸 Seasonal Plant Predictions")
    st.markdown("Predict which plants are likely to **bloom** in a given season and location, based on curated characteristics data.")

    # Season selector
    season = st.selectbox("Select Season", ["Spring", "Summer", "Fall", "Winter"])

    # Optional filters
    with st.expander("⚙️ Advanced Filters"):
        drought = st.selectbox("Drought Tolerance", ["Any", "Low", "Medium", "High"])
        growth = st.selectbox("Growth Habit", ["Any", "Forb/herb", "Shrub", "Tree", "Graminoid", "Vine"])
        native = st.selectbox("Native Status", ["Any", "Native", "Introduced", "Unknown"])

    # Location input
    st.subheader("📍 Filter by Location")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=37.77, format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=-122.42, format="%.4f")
    radius_km = st.slider("Search Radius (km)", 10, 500, 100)

    # ----------------------------
    # Query logic
    # ----------------------------
    if st.button("🔍 Predict Blooming Plants"):
        conn = get_connection()

        # Base query
        query = f"""
            SELECT SCIENTIFICNAME, VERBATIMSCIENTIFICNAME, BLOOMPERIOD, FLOWERCOLOR,
                DROUGHTTOLERANCE, GROWTHHABIT, NATIVESTATUS,
                DECIMALLATITUDE, DECIMALLONGITUDE
            FROM ARQ.PUBLIC.OBSERVATIONSWITHCHARACTERISTICS
            WHERE BLOOMPERIOD ILIKE '%{season}%'
            AND ABS(DECIMALLATITUDE - {lat}) <= {radius_km}/111
            AND ABS(DECIMALLONGITUDE - {lon}) <= {radius_km}/85
        """

        # Add optional filters
        if drought != "Any":
            query += f" AND DROUGHTTOLERANCE ILIKE '{drought}%'"
        if growth != "Any":
            query += f" AND GROWTHHABIT ILIKE '{growth}%'"
        if native != "Any":
            query += f" AND NATIVESTATUS ILIKE '{native}%'"

        # Run query
        df = pd.read_sql(query, conn)

        if df.empty:
            st.warning("No plants found matching these criteria.")
        else:
            st.success(f"✅ Found {len(df)} plants predicted to bloom in **{season}**.")

            # Display result
            st.dataframe(df)

            # Map
            if "DECIMALLATITUDE" in df.columns and "DECIMALLONGITUDE" in df.columns:
                st.map(df.rename(columns={
                    "DECIMALLATITUDE": "lat",
                    "DECIMALLONGITUDE": "lon"
                }))







# ----- Streamlit UI -----




# ---------------- MODE: Home ----------------
if mode == "Home":
    st.markdown(
    "<h1 style='text-align: center; font-size: 60px;'>🌱 <span style='color: orange;'>AR</span>bor<span style='color: white;'>Quest</span></h1>",
    unsafe_allow_html=True
    )

    st.subheader("Explore Nature with AI + LLM + Satellite Data")

    st.markdown("""
    ArborQuest is an AI-powered plant exploration app combining:
    
    - 🌿 **Large Language Models** (Gemini)
    - 🌍 **Geolocation-aware Plant Knowledge**
    - 📷 **AI Image Plant Identification**
    - 🗺️ **Scavenger Hunt Quest Mode**
    
    Enjoy discovering the world of plants around you!
    """)

    # Animated GIF or moving image
    st.video("78665614-cf7b-4b3d-8acc-68624c6999c5.mp4.mp4")

    st.markdown("""
    1. **LLM Plant Q&A:**  
       Ask natural language questions about plants near you, like *"What flowers bloom here in August?"* and get detailed answers.
    
    2. **Plant.id Image ID:**  
       Upload a photo of a plant leaf or flower, and the app will identify it using AI image recognition.

    3. **ArborQuest Quest Mode:**  
       - Choose your difficulty and radius to find local plants for your quest.  
       - Visit the locations shown on the map and look for the plants.  
       - Mark plants as observed and take notes about your findings.  
       - Complete the quest and enjoy your exploration!
    """)

    st.markdown("### Tips for Using Quest Mode")

    st.markdown("""
    - Use the map to navigate to plant observation points within your chosen radius.  
    - Take photos in the field to confirm your finds or for personal reference.  
    - Use the notes section to jot down interesting facts or observations.  
    - Have fun discovering new species and learning more about your local ecosystem!
    """)

    # Smaller images with bigger captions
    def display_image_with_caption(image_path, caption):
        st.image(image_path, width=400)  # Set fixed smaller width (adjust as needed)
        st.markdown(f"<p style='font-size:18px; font-weight:bold; text-align:center;'>{caption}</p>", unsafe_allow_html=True)

    display_image_with_caption("pexels-matthew-montrone-230847-1179229.jpg", "Explore local plants with Quest Mode")
    display_image_with_caption("pexels-marc-nesen-2153115757-33076772.jpg", "Upload photos for plant identification")
    display_image_with_caption("pexels-loquellano-17087539.jpg", "Get detailed answers to your plant questions")

# ---------------- MODE: LLM Q&A ----------------
if mode == "LLM Plant Q&A":
    st.header("🧠 Ask a Question About Local Plants")
    user_query = st.text_area("Ask something like:", "What plants bloom here in August?", height=120)

    plant_query = """
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
    LIMIT 100
    """

    df = pd.read_sql(plant_query, conn)

    if not df.empty and user_query.strip():
        st.info("Querying local plant knowledge...")

        triples = []
        for _, row in df.iterrows():
            species = row["SCIENTIFICNAME"]
            location = row["LOCALITY"] or f"{lat},{lon}"
            date = str(row["EVENTDATE"])[:10]
            triples.append((species, "OBSERVED_AT", location))
            triples.append((species, "BLOOMS_IN", date))
            for k in ["FLOWERCOLOR", "FRUITCOLOR", "GROWTHHABIT", "LIFESPAN", "DURATION", "FAMILY"]:
                if row[k]:
                    triples.append((species, f"HAS_{k}", row[k]))

        kg_text = "\n".join([f"{s} {p} {o}" for s, p, o in triples])
        llm_prompt = f"""
You are a helpful plant expert. A user has asked a question about local flora.

Knowledge Graph:
{kg_text}

User Question: "{user_query}"

Based on this, give a short, accurate answer about plant types, bloom periods, colors, etc.
"""

        with st.spinner("Asking Gemini..."):
    # Use a valid model name from your list here
            model = genai.GenerativeModel("models/gemini-2.5-pro")  
            response = model.generate_content(llm_prompt)
            llm_answer = response.text.strip()
            st.markdown("### 🌸 Answer:")
            st.markdown(llm_answer)

# ---------------- MODE: Plant.id Image ID ----------------
elif mode == "Plant.id Image ID":
    st.header("📷 Identify a Plant with Plant.id")

    uploaded_file = st.file_uploader("Upload a plant photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        st.info("Sending to Plant.id...")

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
        if response.status_code == 200:
            result = response.json()
            suggestions = result.get("suggestions", [])
            if suggestions:
                st.success("🌿 Identification Results:")
                for s in suggestions:
                    st.markdown(f"### 🔎 {s['plant_name']}")
                    st.markdown(f"- **Probability:** {s['probability']:.2%}")
                    if s.get("common_names"):
                        st.markdown(f"- **Common names:** {', '.join(s['common_names'])}")
                    if s.get("similar_images"):
                        st.image(s['similar_images'][0]['url'], caption="Similar Image", width=250)
                    st.markdown("---")
            else:
                st.warning("No match found.")
        else:
            st.error(f"API error: {response.status_code} – {response.text}")



# ---------------- MODE: ArborQuest ----------------
elif mode == "ArborQuest Quest Mode":
    st.header("🌲 ArborQuest – Nature Scavenger Hunt")

    # Sidebar UI
    dl = st.sidebar.selectbox("Choose your difficulty level:", ("Beginner", "Novice", "Advanced"), index=0)
    photo_mode = st.sidebar.checkbox("Enable Photo Mode") if dl == "Beginner" else False
    radius_km = st.sidebar.number_input("Radius (km)", min_value=1, max_value=1000, value=radius_km)

    if photo_mode:
        num_species = 1
        st.sidebar.info("Photo Mode limits the quest to 1 species.")
    else:
        num_species = st.sidebar.number_input("How many species for your quest?", step=1, min_value=1, max_value=25, value=2)

    # Map
    st.subheader("Your Starting Location")
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=10)

    # Cached fetch
    @st.cache_data
    def load_quest_data(lat, lon, radius_km, num_species, photo_mode):
        if photo_mode:
            photo_filter_sql = "AND gbif.PHOTO_URL IS NOT NULL AND UPPER(gbif.PHOTO_URL) != 'NO_PHOTO'"
        else:
            photo_filter_sql = ""

        query = f"""
        SELECT
            o.SCIENTIFICNAME AS "Scientific Name",
            o.COMMON_NAME AS "Common Name",
            o.LAT AS "Latitude",
            o.LON AS "Longitude",
            gbif.PHOTO_URL,
            charac.FAMILY,
            charac.GROWTHHABIT,
            charac.LIFESPAN,
            charac.DURATION,
            charac.FLOWERCOLOR,
            charac.FRUITCOLOR
        FROM ARQ.SAM_AVRAMOV_TEST.OBSERVATION o
        JOIN ARQ.SOURCE.GBIF_OBSERVATION gbif ON o.GBIFID = gbif.GBIFID
        LEFT JOIN ARQ.SOURCE.PLANT_CHARACTERISTICS charac
            ON UPPER(gbif.VERBATIMSCIENTIFICNAME) = UPPER(charac.SCIENTIFICNAME)
        WHERE
            o.COMMON_NAME IS NOT NULL
            {photo_filter_sql}
            AND (
                6371 * 2 * ASIN(
                    SQRT(
                        POWER(SIN(RADIANS(o.LAT - {lat}) / 2), 2) +
                        COS(RADIANS({lat})) * COS(RADIANS(o.LAT)) *
                        POWER(SIN(RADIANS(o.LON - {lon}) / 2), 2)
                    )
                )
            ) <= {radius_km}
        ORDER BY RANDOM()
        LIMIT {num_species}
        """

        return pd.read_sql(query, conn)



    # State management
    questable_species = load_quest_data(lat, lon, radius_km, num_species, photo_mode)
    if "observed_species" not in st.session_state:
        st.session_state.observed_species = set()
    if "notes" not in st.session_state:
        st.session_state.notes = {}

    st.header("Your Quest Checklist")
    if questable_species.empty:
        st.warning("No species found. Try changing location or radius.")
    else:
        cols = st.columns(2)
        quest_sci_names = set(questable_species["Scientific Name"].unique())
        for i, row in questable_species.iterrows():
            sci_name = row["Scientific Name"]
            with cols[i % 2]:
                st.subheader(f"{row['Common Name']}" if dl == "Beginner" else sci_name)
                if dl == "Beginner" and photo_mode:
                    if row['PHOTO_URL'] and row['PHOTO_URL'].strip().upper() != 'NO_PHOTO':
                        st.image(row['PHOTO_URL'], caption=f"Photo of {sci_name}")
                st.info(f"Found near: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
                if dl in ['Beginner', 'Novice']:
                    with st.expander("View Characteristics"):
                        st.markdown(f"**Family:** `{row.get('FAMILY', 'N/A')}`")
                        st.markdown(f"**Growth Habit:** `{row.get('GROWTHHABIT', 'N/A')}`")
                        st.markdown(f"**Lifespan:** `{row.get('LIFESPAN', 'N/A')}`")
                        st.markdown(f"**Flower Color:** `{row.get('FLOWERCOLOR', 'N/A')}`")
                        st.markdown(f"**Fruit Color:** `{row.get('FRUITCOLOR', 'N/A')}`")
                        st.markdown(f"**Flowering Lifespan:** `{row.get('DURATION', 'N/A')}`")

                checked = sci_name in st.session_state.observed_species
                new_checked = st.checkbox("Mark as Observed", value=checked, key=f"chk_{sci_name}")
                if new_checked:
                    st.session_state.observed_species.add(sci_name)
                else:
                    st.session_state.observed_species.discard(sci_name)

                default_note = st.session_state.notes.get(sci_name, "")
                note = st.text_area(f"📝 Notes for {sci_name}", value=default_note, key=f"note_{sci_name}")
                st.session_state.notes[sci_name] = note

                st.markdown("---")

        # Sidebar progress
        observed_in_quest = st.session_state.observed_species.intersection(quest_sci_names)
        st.sidebar.subheader("Quest Progress")
        progress = len(observed_in_quest) / len(quest_sci_names)
        st.sidebar.progress(progress)
        st.sidebar.write(f"{len(observed_in_quest)} of {len(quest_sci_names)} species found.")

        if observed_in_quest:
            st.sidebar.markdown("### Observed:")
            for s_name in sorted(observed_in_quest):
                name_to_list = s_name
                if dl == "Beginner":
                    row = questable_species[questable_species['Scientific Name'] == s_name].iloc[0]
                    name_to_list = row['Common Name']
                st.sidebar.markdown(f"- {name_to_list}")

        if quest_sci_names and len(observed_in_quest) == len(quest_sci_names):
            st.balloons()
            st.success("🎉 Quest Completed!")
            if st.button("Start a New Quest"):
                st.cache_data.clear()
                st.session_state.observed_species.clear()
                st.session_state.notes.clear()
                st.rerun()

    # Notes section
    st.header("Combined Notes")
    notes_data = [{"Species": s, "Notes": n} for s, n in st.session_state.notes.items() if n.strip()]
    if notes_data:
        st.dataframe(pd.DataFrame(notes_data), use_container_width=True)
    else:
        st.info("No notes have been taken yet.")
