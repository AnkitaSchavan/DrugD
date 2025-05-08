import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs, Lipinski, Crippen
from rdkit.Chem import PandasTools, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
import pubchempy as pcp
import py3Dmol
import requests
from io import StringIO, BytesIO
import base64
import time
from functools import lru_cache
from st_aggrid import AgGrid, GridOptionsBuilder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mordred import Calculator, descriptors
import seaborn as sns
import plotly.express as px
from PIL import Image
import json
import io
import traceback
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Page configuration with enhanced theme
st.set_page_config(
    layout="wide",
    page_title="Drug Candidate Exploration Hub",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive pink/drug theme
st.markdown("""
<style>
    /* Background Image */
    body {
        background-image: url('https://images.unsplash.com/photo-1581090700227-1e8e2f6e34b0');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #4A2E4C;
        line-height: 1.6;
    }

    /* Ensure content is readable on top of the background */
    .stApp {
        background: rgba(255, 255, 255, 0.85);
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Header styling */
    .header {
        background: linear-gradient(135deg, #FFD6E8 0%, #FFB6C1 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Card styling */
    .card {
        background: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
        border: 1px solid #E0B0C0;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #E53980 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #E53980 0%, #FF6B6B 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 230, 240, 0.9);
        border-right: 1px solid #E0B0C0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    /* Metric styling */
    .metric {
        background: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E0B0C0;
    }

    /* Progress bar styling */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #FF6B6B 0%, #E53980 100%);
    }

    /* Custom molecule card */
    .molecule-card {
        background: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid #E0B0C0;
    }

    .molecule-card h4 {
        color: #E53980;
    }

    /* Property badge */
    .property-badge {
        display: inline-block;
        background: #FFD1DC;
        color: #4A2E4C;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E0B0C0;
    }

    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: #FFD1DC;
        color: #4A2E4C;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        border: 1px solid #E0B0C0;
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #E53980 100%);
        color: white;
        border-color: #E53980;
    }

    /* Themed horizontal rule */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #E53980 0%, #FFD1DC 50%, #E53980 100%);
        margin: 1.5rem 0;
    }

    /* Themed titles */
    h1, h2, h3, h4, h5, h6 {
        color: #E53980;
    }

    /* Themed text */
    p, label, .stMarkdown {
        color: #4A2E4C;
    }

    /* Info/Success/Error messages */
    .stAlert {
        background-color: #FFF0F5;
        color: #4A2E4C;
        border-color: #E0B0C0;
    }

    .stSuccess {
        background-color: #F0FFF0;
        color: #4A2E4C;
        border-color: #C1E1C1;
    }

    .stError {
        background-color: #FFF0F5;
        color: #E53980;
        border-color: #E53980;
    }

    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background-image: url('https://images.unsplash.com/photo-1581090700227-1e8e2f6e34b0');
        background-size: cover;
        background-position: center;
        color: #4A2E4C;
        padding: 1.5rem;
        border-right: 2px solid #E0B0C0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }

    /* Equal length radio buttons */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #E0B0C0;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        min-height: 3.5rem;
        color: #4A2E4C;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #FFD6E8;
        cursor: pointer;
    }

    [data-testid="stSidebar"] .stRadio input:checked + div > label {
        background: linear-gradient(135deg, #FF6B6B, #E53980);
        color: white !important;
        font-weight: 900;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    /* Sidebar header */
    .sidebar-header {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        font-family: 'Poppins', sans-serif;
        background-image: url('https://images.unsplash.com/photo-1581090700227-1e8e2f6e34b0');
        background-size: cover;
        background-position: center;
        color: #fff;
    }
    .sidebar-header h2 {
        color: #ffffff;
        text-shadow: 1px 1px 4px #000000;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 0.3rem;
    }
    .sidebar-header p {
        color: #fddde6;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_mol' not in st.session_state:
    st.session_state.selected_mol = None
if 'selected_smiles' not in st.session_state:
    st.session_state.selected_smiles = None
if 'last_selected_db' not in st.session_state:
    st.session_state.last_selected_db = {}
if 'generate_viz' not in st.session_state:
    st.session_state.generate_viz = False
if 'smiles_input' not in st.session_state:
    st.session_state.smiles_input = {}

# Sidebar navigation
with st.sidebar:
    app_mode = st.radio(
        "Select Module:",
        options=[
            "ℹ️ Home page",
            "🏠 Dashboard Overview",
            "🔍 Drug Explorer",
            "🧮 Molecular Property Calculator",
            "📊 Advanced Similarity Search",
            "💊 ADMET Prediction",
            "🧩 Scaffold Analysis",
            "🖥️ Virtual Screening",
            "⚗️ Compound Optimization",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-top: 2px dashed #E0B0C0;'>", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner="Loading drug dataset...")
def download_dataset():
    SOURCES = [
        "https://www.cureffi.org/wp-content/uploads/2013/10/drugs.txt",
        "https://raw.githubusercontent.com/your-repo/backup/main/drugs.txt" # Add a backup source if the first fails
    ]

    for url in SOURCES:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), sep="\t")

            # Basic cleaning
            df = df[['generic_name', 'smiles']].dropna()

            # Validate SMILES
            df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
            df = df[df['mol'].notna()].copy()

            # Calculate properties
            def calculate_properties(mol):
                return {
                    'mw': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Lipinski.NumHDonors(mol),
                    'hba': Lipinski.NumHAcceptors(mol),
                    'rot_bonds': Lipinski.NumRotatableBonds(mol),
                    'tpsa': Descriptors.TPSA(mol)
                }

            props = df['mol'].apply(calculate_properties).apply(pd.Series)
            df = pd.concat([df, props], axis=1)

            # Apply drug-like filters
            druglike_filter = (
                (df['mw'].between(150, 600)) &
                (df['logp'].between(-2, 5)) &
                (df['hbd'] <= 5) &
                (df['hba'] <= 10) &
                (df['rot_bonds'] <= 10) &
                (df['tpsa'] <= 150)
            )

            return df[druglike_filter].drop(columns='mol').reset_index(drop=True)

        except Exception as e:
            st.warning(f"Failed to load from {url}: {str(e)}")
            continue

    st.error("All data sources failed. Using sample data.")
    return pd.DataFrame({
        'generic_name': ['Aspirin', 'Paracetamol'],
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(=O)NC1=CC=C(C=C1)O'],
        'mw': [180.16, 151.16],
        'logp': [1.19, 0.49],
        'hbd': [1, 2],
        'hba': [3, 2],
        'rot_bonds': [2, 1],
        'tpsa': [63.6, 49.3]
    })

# Load data at the beginning
data = download_dataset()

# --- Function to display molecule input section ---
def display_molecule_input():
    st.markdown("### 🔬 Molecule Input")
    input_method = st.radio(
        "Input Method:",
        ["Select from Database", "Enter SMILES"],
        label_visibility="visible",
        key=f"input_method_{app_mode}" # Unique key per page
    )

    col1, col2 = st.columns([1, 2]) # Use columns for layout

    with col1:
        if input_method == "Select from Database":
            compound_name = st.selectbox(
                "Select compound:",
                options=data['generic_name'].unique(),
                index=0,
                key=f'db_select_{app_mode}' # Unique key per page
            )

            # Update session state only if the selected compound changes
            if st.session_state.get(f'last_selected_db_{app_mode}') != compound_name:
                selected = data[data['generic_name'] == compound_name].iloc[0]
                st.session_state.selected_smiles = selected['smiles']
                st.session_state.selected_mol = Chem.MolFromSmiles(st.session_state.selected_smiles)
                st.session_state[f'last_selected_db_{app_mode}'] = compound_name
                # Clear other input method's state if needed
                st.session_state[f'smiles_input_{app_mode}'] = '' # Reset SMILES input field

            #


        elif input_method == "Enter SMILES":
            smiles_input = st.text_input("Enter SMILES:",
                                         value=st.session_state.get(f'smiles_input_{app_mode}', ''), # Retain value
                                         key=f"smiles_input_{app_mode}") # Unique key per page
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    st.success("✅ Valid SMILES")
                    img = Draw.MolToImage(mol, size=(200, 200))
                    st.image(img, caption="Input Molecule", use_container_width=True)
                    st.session_state.selected_mol = mol
                    st.session_state.selected_smiles = smiles_input
                    # Clear DB selection state
                    st.session_state[f'last_selected_db_{app_mode}'] = None
                else:
                    st.error("❌ Invalid SMILES")
                    st.session_state.selected_mol = None # Clear molecule if invalid
                    st.session_state.selected_smiles = None

    with col2:
        if st.session_state.selected_mol:
             st.markdown("""
            <div class="molecule-card">
                <h4 style="margin-top:0;">Selected Molecule</h4>
                <p><strong>SMILES:</strong> {}</p>
                <p><strong>MW:</strong> {:.2f}</p>
                <p><strong>LogP:</strong> {:.2f}</p>
            </div>
            """.format(st.session_state.selected_smiles,
                       Descriptors.MolWt(st.session_state.selected_mol),
                       Descriptors.MolLogP(st.session_state.selected_mol)), unsafe_allow_html=True)
        else:
             st.info("Select or enter a molecule to proceed.")

    st.markdown("---") # Separator
    return input_method

# Get current molecule from session state
mol = st.session_state.selected_mol
smiles_input = st.session_state.selected_smiles

# --- Place the function before your main code ---
def get_database_fps(data, fp_type):
    fps = []
    mols = []
    names = []

    for _, row in data.iterrows():
        smi = row['smiles']
        name = row.get('generic_name', smi)  # fallback to SMILES if no name

        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            names.append(name)

            try:
                if fp_type == "Morgan":
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                elif fp_type == "MACCS":
                    fp = AllChem.GetMACCSKeysFingerprint(mol)
                elif fp_type == "RDKit":
                    fp = Chem.RDKFingerprint(mol)
                else:
                    fp = None

                fps.append(fp)
            except:
                fps.append(None)
        else:
            fps.append(None)

    return fps, mols, names

# --- Then your app code ---

# --- Main App Logic ---
if app_mode == "ℹ️ Home page":
    st.markdown("""
    <div class="header" style="color:black;">
        <h1 style="color:black; margin:0;">🧪 Drug Candidate Exploration Hub</h1>
        <p style="color:black; margin:0; opacity:0.8;">Comprehensive cheminformatics platform for modern drug discovery</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Platform Overview Section
    st.markdown("""
    ## 🌟 Platform Overview
    
    The **Drug Discovery Suite** is a state-of-the-art cheminformatics platform designed to accelerate small molecule drug discovery through:
    
    - **Interactive molecular analysis** with real-time visualization
    - **Comprehensive property calculation** with 200+ descriptors
    - **Advanced predictive modeling** for ADMET properties
    - **Target identification** through ChEMBL integration
    - **Virtual screening** capabilities for large compound libraries
    
    Built on RDKit and Streamlit, this platform combines robust computational chemistry with an intuitive interface.
    """)

    # Key Features Section
    st.markdown("""
    ## 🚀 Key Capabilities
    
    ### Core Modules:
    
    | Module | Description | Key Features |
    |--------|-------------|--------------|
    | **Dashboard** | Overview of compound collections | Property distributions, chemical space visualization |
    | **Drug Explorer** | Browse and filter compound databases | Advanced filtering, scaffold analysis |
    | **Property Calculator** | Calculate molecular descriptors | Basic and advanced descriptors, 2D/3D visualization,Fingerprints |
    | **Similarity Search** | Find structurally similar compounds |similar compound based on  Multiple fingerprint , batch processing |
    | **ADMET Prediction** | Predict drug-like properties | Drug likeness ,Lipinski rules, toxicity alerts, BBB penetration |
    | **Virtual Screening** | Screen compound libraries | Customizable thresholds, structural alerts |
    | **Compound Optimization** | Improve drug properties | Bioisostere suggestions, property optimization |
    
    """)

    # User Guide Section
    st.markdown("""
    ## 📖 User Guide
    
    ### Getting Started
    
    1. **Select a Molecule**:
       - Choose from the built-in database of drug-like compounds
       - Or enter a SMILES string directly
       - View basic properties in the Dashboard
    
    2. **Explore Features**:
       - Calculate descriptors in Molecular Property Calculator
       - Run similarity searches against the database
       - Predict ADMET properties
       - Perform virtual screening
    
    3. **Advanced Workflows**:
       - Upload your own compound libraries (CSV/TXT)
       - Analyze scaffold networks
       - Get optimization suggestions
    
    ### Best Practices
    
    - **For Property Calculation**:
      - Start with the Basic Properties tab for key descriptors
      - Use Advanced Descriptors for specialized calculations
      - Check the 2D/3D Viewer to validate structures
    
    - **For Similarity Searching**:
      - Morgan fingerprints (radius=2) work well for general similarity
      - MACCS keys are better for scaffold hopping
      - Adjust similarity threshold based on your needs (0.7-0.9 typical)
    
    - **For Virtual Screening**:
      - Check structural alerts first (PAINS, Brenk filters)
      - Start with moderate similarity thresholds (0.6-0.7)
      - Consider multiple fingerprint types for comprehensive results
    
    - **For Compound Optimization**:
      - Address Lipinski rule violations first
      - Consider bioisosteric replacements for problematic groups
      - Balance solubility and permeability properties
    """)

elif app_mode == "🏠 Dashboard Overview":
    # Dashboard header
    
    st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, #7F7FD5, #86A8E7, #91EAE4);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric {
        background: white;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .metric h2 {
        margin: 0.2rem 0;
        font-weight: 700;
    }
    </style>

    <div class="header" style="color:black;">
        <h1 style="color:black; margin:0;"> 🏠 Dashboard Overview </h1>
        <p style="color:black; margin:0; opacity:0.8;">Comprehensive cheminformatics platform for modern drug discovery</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input()

    # Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric">
            <h3>Total Compounds</h3>
            <h2 style="color:#2C3E50;">{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_mw = round(data['mw'].mean(), 2)
        st.markdown(f"""
        <div class="metric">
            <h3>Avg MW</h3>
            <h2 style="color:#2C3E50;">{avg_mw} Da</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_logp = round(data['logp'].mean(), 2)
        st.markdown(f"""
        <div class="metric">
            <h3>Avg LogP</h3>
            <h2 style="color:#2C3E50;">{avg_logp}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric">
            <h3>Drug-like</h3>
            <h2 style="color:#2C3E50;">100%</h2>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.markdown("## 📈 Molecular Property Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Molecular Weight")
        fig1 = px.histogram(data, x='mw', nbins=30, color_discrete_sequence=['#6B73FF'])
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

        # Download MW Histogram
        csv1 = data[['mw']].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download MW Data", data=csv1, file_name='molecular_weight.csv', mime='text/csv')

    with col2:
        st.markdown("#### LogP Distribution")
        fig2 = px.box(data, y='logp', color_discrete_sequence=['#000DFF'])
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

        # Download LogP
        csv2 = data[['logp']].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download LogP Data", data=csv2, file_name='logp_distribution.csv', mime='text/csv')

    # Relationship Plots
    st.markdown("## 🔍 Property Relationships")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### MW vs LogP (Colored by TPSA, Size by Rotatable Bonds)")
        fig3 = px.scatter(data, x='mw', y='logp', color='tpsa', size='rot_bonds',
                          color_continuous_scale='bluered')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

        csv3 = data[['mw', 'logp', 'tpsa', 'rot_bonds']].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download MW vs LogP Data", data=csv3, file_name='mw_vs_logp.csv', mime='text/csv')

    with col2:
        st.markdown("#### Property Correlations")
        corr_matrix = data[['mw', 'logp', 'hbd', 'hba', 'tpsa']].corr()
        fig4 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='bluered')
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

        csv4 = corr_matrix.to_csv().encode('utf-8')
        st.download_button("⬇️ Download Correlation Matrix", data=csv4, file_name='property_correlation.csv', mime='text/csv')


elif app_mode == "🔍 Drug Explorer":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">🔍 Drug Explorer</h1>
        <p style="color:black; margin:0; opacity:0.8;">Browse and filter the compound database</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input() # Add molecule input section

    st.markdown("## 🗃️ Compound Database")

    with st.expander("🔍 Advanced Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            mw_range = st.slider(
                "Molecular Weight (Da)",
                min_value=int(data['mw'].min()),
                max_value=int(data['mw'].max()),
                value=(150, 500),
                step=10,
                key=f'mw_range_{app_mode}'
            )
            logp_range = st.slider(
                "LogP Range",
                min_value=float(data['logp'].min()),
                max_value=float(data['logp'].max()),
                value=(-1.0, 3.0),
                step=0.1,
                key=f'logp_range_{app_mode}'
            )
        with col2:
            tpsa_range = st.slider(
                "TPSA (Å²)",
                min_value=int(data['tpsa'].min()),
                max_value=int(data['tpsa'].max()),
                value=(20, 120),
                step=5,
                key=f'tpsa_range_{app_mode}'
            )
            rot_bonds = st.slider(
                "Max Rotatable Bonds",
                min_value=0,
                max_value=int(data['rot_bonds'].max()),
                value=5,
                key=f'rot_bonds_{app_mode}'
            )

    # Apply filters
    filtered_data = data[
        (data['mw'].between(*mw_range)) &
        (data['logp'].between(*logp_range)) &
        (data['tpsa'].between(*tpsa_range)) &
        (data['rot_bonds'] <= rot_bonds)
    ]

    # Display filtered data
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top:0;">🔬 Found {len(filtered_data)} matching compounds</h3>
    </div>
    """, unsafe_allow_html=True)

    # Configure AgGrid with custom styling
    gb = GridOptionsBuilder.from_dataframe(filtered_data)
    gb.configure_default_column(
        filterable=True,
        sortable=True,
        resizable=True,
        editable=False,
        wrapText=True,
        autoHeight=True
    )
    gb.configure_column("smiles", headerName="SMILES", width=300)
    gb.configure_grid_options(domLayout='normal')
    grid_options = gb.build()

    AgGrid(
        filtered_data,
        gridOptions=grid_options,
        height=400,
        width='100%',
        theme='streamlit',
        fit_columns_on_grid_load=False,
        custom_css={
            ".ag-header-cell-label": {"justify-content": "center"},
            ".ag-cell": {"display": "flex", "align-items": "center"}
        },
        key=f'aggrid_{app_mode}' # Unique key per page
    )

    # Visualization
    st.markdown("## 📈 Chemical Space Analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        viz_type = st.radio(
            "Visualization Method:",
            ["PCA", "t-SNE"],
            horizontal=True,
            help="PCA shows global trends, t-SNE shows local clusters",
            key=f'viz_type_{app_mode}'
        )
    with col2:
        if st.button("🔄 Generate Visualization", use_container_width=True, key=f'generate_viz_button_{app_mode}'):
            st.session_state.generate_viz = True

    if st.session_state.get('generate_viz', False) and not filtered_data.empty:
        with st.spinner("Computing visualization..."):
            # Prepare features
            features = filtered_data[['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rot_bonds']]

            # Reduce dimensions
            try:
                if viz_type == "PCA":
                    reducer = PCA(n_components=2)
                else:
                    # Ensure perplexity is less than the number of samples
                    perplexity_val = min(30, len(filtered_data) - 1)
                    if perplexity_val <= 1:
                         st.warning("Not enough data points for t-SNE. Need at least 2.")
                         st.session_state.generate_viz = False
                         st.stop()
                    reducer = TSNE(n_components=2, perplexity=perplexity_val)

                embedding = reducer.fit_transform(features)

                # Create interactive plot
                fig = px.scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    color=filtered_data['logp'],
                    size=filtered_data['mw']/100,
                    hover_name=filtered_data['generic_name'],
                    color_continuous_scale='bluered',
                    labels={'color': 'LogP', 'size': 'MW'},
                    title=f"{viz_type} Projection of Chemical Space"
                )

                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=f"{viz_type}-1",
                    yaxis_title=f"{viz_type}-2"
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")
                st.session_state.generate_viz = False # Reset flag on failure


elif app_mode == "🧮 Molecular Property Calculator":
    st.markdown("""
    <style>
        /* Increase font size for tabs */
        .streamlit-expanderHeader {
            font-size: 20px;
            font-weight: bold;
        }

        /* Adjust general header */
        .header h1 {
            font-size: 30px;
            font-weight: bold;
        }

        .header p {
            font-size: 16px;
            opacity: 0.8;
        }

        /* Custom styling for the properties section */
        .property-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .property-card .title {
            font-size: 18px;
            font-weight: bold;
        }

        .property-card .value {
            font-size: 22px;
            color: #007bff;
        }

        .property-card .unit {
            font-size: 16px;
            color: #6c757d;
        }
        

        .section-title {
            font-size: 2.5rem; /* increase for better readability */
        }

        .profile-card p {
            font-size: 2.0rem; /* increase paragraph text size */
        }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">🧮 Molecular Property Calculator</h1>
        <p style="color:black; margin:0; opacity:0.8;">Calculate various physicochemical descriptors</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input()  # Add molecule input section

    if not mol:
        st.warning("No molecule selected")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Properties", "📊 Advanced Descriptors", "🖼️ 2D/3D Viewer", "🔢 Fingerprints"])

    # Tab 1: Basic Properties
    with tab1:
        st.write("### Molecular Properties")

        # Add properties in a visually attractive way
        props = [
            ("Molecular Weight", f"{Descriptors.MolWt(mol):.2f}", "g/mol", "⚖️"),
            ("Exact Mass", f"{Descriptors.ExactMolWt(mol):.4f}", "", "🧮"),
            ("LogP", f"{Descriptors.MolLogP(mol):.2f}", "", "📈"),
            ("TPSA", f"{Descriptors.TPSA(mol):.2f}", "Å²", "📐"),
            ("H-Bond Donors", Lipinski.NumHDonors(mol), "", "💧"),
            ("H-Bond Acceptors", Lipinski.NumHAcceptors(mol), "", "💦"),
            ("Rotatable Bonds", Lipinski.NumRotatableBonds(mol), "", "🔄"),
            ("Ring Count", Lipinski.RingCount(mol), "", "⭕"),
            ("Aromatic Rings", Lipinski.NumAromaticRings(mol), "", "🔴"),
            ("Fraction CSP3", f"{Lipinski.FractionCSP3(mol):.2f}", "", "🔶"),
            ("Formal Charge", Chem.GetFormalCharge(mol), "", "⚡")
        ]

        # Loop through the properties and display them in styled cards
        for i, (name, value, unit, icon) in enumerate(props):
            st.markdown(f"""
                <div class="property-card">
                    <div class="title">{icon} {name}</div>
                    <div class="value">{value} {unit}</div>
                    <div class="unit">{unit}</div>
                </div>
            """, unsafe_allow_html=True)

    # Tab 2: Advanced Descriptors
    with tab2:
        st.write("### Advanced Descriptors")
        try:
            if not hasattr(np, 'float'):
                np.float = float

            calc = Calculator(descriptors)
            try:
                desc_values = calc(mol)
                desc_data = {str(k): str(v) for k, v in desc_values.items() if v is not None}
                desc_df = pd.DataFrame.from_dict(desc_data, orient='index', columns=['Value'])
            except Exception as calc_e:
                 st.error(f"Error during descriptor calculation: {str(calc_e)}")
                 desc_df = pd.DataFrame(columns=['Value'])

            desc_class = st.selectbox("Filter by:", ["All", "Topological", "Geometrical", "Electronic", "Constitutional"], key=f'desc_filter_{app_mode}')
            if desc_class != "All":
                desc_df = desc_df[desc_df.index.str.lower().str.startswith(desc_class[:3].lower(), na=False)]

            st.dataframe(desc_df, use_container_width=True, height=400)

        except Exception as e:
            st.error(f"Descriptor calculation setup failed: {str(e)}")

    # Tab 3: 2D and 3D Viewer
    with tab3:
        st.write("### 2D and 3D Molecular Viewer")
        st.write("Visualize the 2D structure and an estimated 3D conformation.")

        # Ensure session state variable is initialized
        if 'last_selected_name' not in st.session_state:
            st.session_state.last_selected_name = None

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### 2D Structure")
            try:
                img = Draw.MolToImage(mol, size=(400, 300))  # 2D image
                st.image(img, use_container_width=True, caption="2D Depiction")
            except Exception as e:
                st.warning(f"Could not generate 2D image: {e}")

        with col2:
            st.write("#### 3D Structure (Estimated)")
            st.caption("Note: 3D conformation is an estimate using UFF optimization and may not represent the lowest energy state.")
            try:
                mol_3d = Chem.AddHs(mol)
                embed_success = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                if embed_success == -1:
                    st.warning("Could not generate 3D coordinates with ETKDG. Trying UFF.")
                    embed_success = AllChem.EmbedMolecule(mol_3d, AllChem.UFF())
                    if embed_success == -1:
                        st.error("Could not generate 3D coordinates for this molecule.")
                        mol_3d = None

                if mol_3d:
                    AllChem.UFFOptimizeMolecule(mol_3d)
                    molblock = Chem.MolToMolBlock(mol_3d)

                    viewer_html = f"""
                        <div id='viewer_3d' style='height:350px;width:100%'></div>
                        <script src='https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'></script>
                        <script>
                        let viewer = $3Dmol.createViewer("viewer_3d", {{backgroundColor:"white"}});
                        viewer.addModel(`{molblock}`, "mol");
                        viewer.setStyle({{}}, {{stick:{{}}}});
                        viewer.zoomTo();
                        viewer.render();
                        </script>
                    """

                    st.components.v1.html(viewer_html, height=370)

                    mol_name = st.session_state.get("last_selected_name", "molecule")
                    st.download_button(
                        label="📥 Download 3D Structure (MolBlock)",
                        data=molblock,
                        file_name=f"{mol_name}_3d.mol",
                        mime="chemical/x-mdl-molfile",
                        key="download_3d_molblock"
                    )

            except Exception as e:
                st.error(f"3D visualization failed: {e}")
                st.write(traceback.format_exc())

    # Tab 4: Fingerprint Calculator (Enhanced)
    with tab4:
        st.write("## 🧬 Molecular Fingerprint Generator")

        # Option selection
        analysis_type = st.radio("Select input type:",
                               ["Single Molecule", "Batch Processing (CSV/TSV)"],
                               horizontal=True,
                               key=f'fp_analysis_type_{app_mode}')
        if analysis_type == "Single Molecule":
            if not mol:
                st.warning("No molecule selected in the main interface")
                st.stop()
                
            st.success(f"Analyzing: {Chem.MolToSmiles(mol)}")
            
            # Fingerprint configuration
            col1, col2 = st.columns(2)
            with col1:
                fp_type = st.selectbox("Fingerprint type",
                                    ["Morgan", "MACCS", "RDKit"],
                                    key="fp_type_single")
                
            with col2:
                if fp_type == "Morgan":
                    radius = st.slider("Radius", 1, 5, 2)
                    n_bits = st.slider("Number of bits", 64, 4096, 1024, step=64)
            
            # Generate fingerprint
            if fp_type == "Morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            elif fp_type == "MACCS":
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == "RDKit":
                fp = Chem.RDKFingerprint(mol)
            
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            # Display results
            st.write("### 🔍 Fingerprint Results")
            st.write(f"**Type:** {fp_type} | **Bits:** {len(arr)} | **Set bits:** {int(sum(arr))}")
            
            # Visualize fingerprint
            fig, ax = plt.subplots(figsize=(10, 1.5))
            ax.imshow(arr.reshape(1, -1), cmap='viridis', aspect='auto')
            ax.set_yticks([])
            ax.set_xlabel("Bit Position")
            st.pyplot(fig)
            
            # Download options
            st.download_button(
                label="📥 Download Fingerprint (CSV)",
                data=pd.DataFrame(arr).to_csv(index=False, header=False).encode('utf-8'),
                file_name=f"fingerprint_{fp_type}.csv",
                mime="text/csv"
            )
        
        else:  # Batch Processing
            st.info("Upload a CSV/TSV file containing SMILES strings")
            
            uploaded_file = st.file_uploader("Choose a file", 
                                           type=['csv', 'tsv'],
                                           key="batch_uploader")
            
            if uploaded_file:
                st.success(f"Processing {uploaded_file.name}")
                
                # Read file
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file, sep='\t')
                    
                    # SMILES column selection
                    smiles_col = st.selectbox("Select SMILES column", 
                                            df.columns,
                                            key="smiles_col")
                    
                    # Fingerprint configuration
                    fp_type = st.selectbox("Fingerprint type",
                                        ["Morgan", "MACCS", "RDKit"],
                                        key="fp_type_batch")
                    
                    if fp_type == "Morgan":
                        radius = st.slider("Radius", 1, 5, 2, key="batch_radius")
                        n_bits = st.slider("Number of bits", 64, 4096, 1024, step=64, key="batch_bits")
                    
                    # Process molecules
                    with st.spinner("Generating fingerprints..."):
                        valid_mols = []
                        fps = []
                        
                        for smi in df[smiles_col]:
                            mol = Chem.MolFromSmiles(str(smi))
                            if mol:
                                valid_mols.append(mol)
                                if fp_type == "Morgan":
                                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                                elif fp_type == "MACCS":
                                    fp = AllChem.GetMACCSKeysFingerprint(mol)
                                elif fp_type == "RDKit":
                                    fp = Chem.RDKFingerprint(mol)
                                
                                arr = np.zeros((1,))
                                DataStructs.ConvertToNumpyArray(fp, arr)
                                fps.append(arr)
                        
                        if not valid_mols:
                            st.error("No valid molecules found in the file!")
                            st.stop()
                            
                        st.success(f"Generated fingerprints for {len(valid_mols)} molecules")
                        
                        # Create results
                        fps_array = np.vstack(fps)
                        results_df = pd.DataFrame(fps_array,
                                                columns=[f"Bit_{i}" for i in range(fps_array.shape[1])],
                                                index=df.index)
                        
                        # Combine with original data
                        output_df = pd.concat([df, results_df], axis=1)
                        
                        # Show preview
                        st.write("### Results Preview")
                        st.dataframe(output_df.head(3))
                        
                        # Download options
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=output_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"batch_fingerprints_{fp_type}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    
                    
# Place this function somewhere above your app logic
                    
        # Fingerprint logic remains unchanged here, with possible enhancements for visual appeal
elif app_mode == "📊 Advanced Similarity Search":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">📊 Advanced Similarity Search</h1>
        <p style="color:black; margin:0; opacity:0.8;">Find similar compounds using molecular fingerprints</p>
    </div>
    """, unsafe_allow_html=True)

    # Input molecule
    smiles_input = display_molecule_input()

    st.markdown('<hr style="height:2px;border-width:0;color:gray;background:linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet)">',
                unsafe_allow_html=True)

    # Fingerprint setting (now only visible in this tab)
    st.markdown("### 🧪 Fingerprint Settings")
    fp_type = st.radio(
        "Select fingerprint type:",
        ["Morgan", "MACCS"],
        horizontal=True,
        help="Morgan: Circular fingerprints | MACCS: Predefined keys",
        key='sim_fp_type'
    )

    # Upload file for batch search
    st.markdown("### 📁 Upload Compounds & Find Similarities in Database")
    uploaded_file = st.file_uploader("Upload compound file", type=["csv", "txt"], key='sim_upload')

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                compounds_df = pd.read_csv(uploaded_file)
            else:
                compounds_df = pd.read_csv(uploaded_file, header=None, names=['SMILES'])
            smiles_list = []
            for _, row in compounds_df.iterrows():
                smi = row['SMILES'] if 'SMILES' in row else row[0]
                smiles_list.append(smi)

            # Convert SMILES to molecules
            molecules = []
            valid_smiles = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    molecules.append(mol)
                    valid_smiles.append(smi)
                else:
                    st.warning(f"Invalid SMILES skipped: {smi}")

            if not molecules:
                st.warning("No valid molecules found.")
            else:
                fps, mols, names = get_database_fps(data, fp_type)
                top_n = 5
                results_list = []

                for idx, mol in enumerate(molecules):
                    query_fp = (
                        AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) if fp_type == "Morgan"
                        else MACCSkeys.GenMACCSKeys(mol)
                    )
                    similarities = []
                    for i, db_fp in enumerate(fps):
                        if db_fp:
                            try:
                                sim = DataStructs.TanimotoSimilarity(query_fp, db_fp)
                                similarities.append((sim, names[i], mols[i], valid_smiles[idx]))
                            except:
                                continue
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_hits = similarities[:top_n]
                    results_list.append({'Input SMILES': valid_smiles[idx], 'Top matches': top_hits})

                # Display results: table + images
                for res in results_list:
                    st.markdown(f"### Input Molecule: {res['Input SMILES']}")
                    df_matches = pd.DataFrame([{
                        'Name': hit[1],
                        'SMILES': hit[3],
                        'Similarity': f"{hit[0]:.3f}"
                    } for hit in res['Top matches']])
                    st.dataframe(df_matches)

                    cols = st.columns(len(res['Top matches']))
                    for col, hit in zip(cols, res['Top matches']):
                        with col:
                            if hit[2]:
                                st.image(Draw.MolToImage(hit[2], size=(200, 200)),
                                         caption=f"{hit[1]}\nSim: {hit[0]:.3f}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        
    # --- Compare two molecules only in Advanced Similarity Search page ---
    st.markdown("### 🔍 Compare Two Molecules")
    col1, col2 = st.columns(2)
    with col1:
        smi1 = st.text_input("SMILES 1")
    with col2:
        smi2 = st.text_input("SMILES 2")
    if st.button("Compare SMILES") and smi1 and smi2:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 and mol2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
            sim_score = DataStructs.TanimotoSimilarity(fp1, fp2)
            # Show molecules side by side
            col_left, col_right = st.columns(2)
            with col_left:
                st.image(Draw.MolToImage(mol1, size=(300, 300)), caption="Molecule 1")
            with col_right:
                st.image(Draw.MolToImage(mol2, size=(300, 300)), caption="Molecule 2")
            # Show similarity score prominently below
            st.markdown(f"### **Similarity score: {sim_score:.3f}**")
        else:
            st.error("Invalid SMILES entered.")

elif app_mode == "💊 ADMET Prediction":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">💊 ADMET Prediction</h1>
        <p style="color:black; margin:0; opacity:0.8;">Predict Absorption, Distribution, Metabolism, Excretion, and Toxicity properties</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input() # Add molecule input section

    if not mol:
        st.warning("No molecule selected")
        st.stop()

    # Pre-calculate properties with error handling
    props = {}
    try:
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rot_bonds': Descriptors.NumRotatableBonds(mol),
            'mr': Crippen.MolMR(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'charge': Chem.GetFormalCharge(mol)
        }
    except Exception as e:
        st.error(f"Error calculating basic properties: {str(e)}")
        st.stop()


    tab1, tab2, tab3, tab4 = st.tabs(["💊 Drug-likeness", "🚀 Absorption", "☠️ Toxicity", "🧠 BBB"])

    with tab1:
        st.write("### Drug-likeness Rules")

        # Define all rules
        rules = {
            "Lipinski's Rule of 5": [
                ("MW < 500", props.get('mw', float('inf')) < 500),
                ("LogP < 5", props.get('logp', float('inf')) < 5),
                ("HBD ≤ 5", props.get('hbd', float('-inf')) <= 5),
                ("HBA ≤ 10", props.get('hba', float('-inf')) <= 10)
            ],
            "Ghose Filter": [
                ("-0.4 < LogP < 5.6", -0.4 < props.get('logp', float('nan')) < 5.6),
                ("160 < MW < 480", 160 < props.get('mw', float('nan')) < 480),
                ("40 < Atoms < 70", 40 < props.get('heavy_atoms', float('nan')) < 70),
                ("MR 40-130", 40 < props.get('mr', float('nan')) < 130)
            ],
            "Veber's Rule": [
                ("Rot. Bonds ≤ 10", props.get('rot_bonds', float('-inf')) <= 10),
                ("TPSA ≤ 140", props.get('tpsa', float('inf')) <= 140)
            ]
        }

        # Display rules with metrics
        total_conditions = sum(len(c) for c in rules.values())
        total_passed = 0
        for rule_name, conditions in rules.items():
            passed = sum(1 for _, cond in conditions if cond)
            total_passed += passed
            st.metric(f"{rule_name} ({passed}/{len(conditions)})",
                     "✅ Passed" if passed == len(conditions) else "⚠️ Review",
                     help="\n".join(f"{name}: {'✔' if cond else '✖'}" for name, cond in conditions))

        # Overall assessment
        score = total_passed / total_conditions if total_conditions > 0 else 0
        st.progress(score,
                    text=f"Overall Drug-likeness: {'Excellent' if score>0.8 else 'Good' if score>0.6 else 'Poor' if score > 0.3 else 'Very Poor'}")


    with tab2:
        st.write("### Absorption Potential")

        # Bioavailability score
        bio_score = sum([
            props.get('mw', float('inf')) < 500,
            -0.4 < props.get('logp', float('nan')) < 5.6,
            props.get('tpsa', float('inf')) < 140,
            props.get('rot_bonds', float('-inf')) <= 10
        ])
        st.metric("Bioavailability Score", f"{bio_score}/4")

        # Solubility prediction (simplified model)
        # Ensure all required properties are available before calculation
        logS = 'N/A'
        if all(prop in props for prop in ['logp', 'mw', 'tpsa', 'rot_bonds']):
             try:
                 logS_val = 0.16 - 0.63*props['logp'] - 0.0062*props['mw'] + 0.066*props['tpsa'] - 0.74*props['rot_bonds']
                 logS = f"{logS_val:.2f}"
                 logS_status = "Good" if logS_val > -4 else "Poor"
             except Exception as e:
                 st.warning(f"Could not calculate LogS: {str(e)}")
                 logS_status = "Calculation Error"
        else:
             logS_status = "Missing Properties"


        st.metric("Predicted LogS", logS, logS_status)


    with tab3:
        st.write("### Toxicity Assessment")

        # Toxicophores check
        toxic_groups = {
            "Sulfonyl halide": "[S;D1](=O)(=O)[Cl,Br,I,F]",
            "Anhydride": "C(=O)OC(=O)",
            "Azo": "N=N",
            "Carbamate": "[NH2]C(=O)O",
            "Sulfonate": "[S;D2]([#6])(=O)(=O)"
        }

        toxic_found = []
        for name, smarts in toxic_groups.items():
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    toxic_found.append(name)
            except Exception as e:
                st.warning(f"Could not check toxicophore {name}: {str(e)}")


        if toxic_found:
            st.error(f"Toxicophores detected: {', '.join(toxic_found)}")
        else:
            st.success("✅ No toxicophores detected")

        # LD50 prediction (simplified model)
        # Ensure all required properties are available before calculation
        ld50 = 'N/A'
        if all(prop in props for prop in ['logp', 'mw', 'tpsa']):
             try:
                 ld50_val = 2.5 - 0.5*props['logp'] - 0.01*props['mw'] + 0.05*props['tpsa']
                 ld50 = f"{ld50_val:.1f}"
                 ld50_status = "Toxicity concern" if ld50_val < 2 else "Normal range"
             except Exception as e:
                 st.warning(f"Could not calculate LD50: {str(e)}")
                 ld50_status = "Calculation Error"
        else:
             ld50_status = "Missing Properties"

        st.metric("Estimated LD50", ld50, ld50_status)

    with tab4:
        st.write("### Blood-Brain Barrier")

        # Ensure all required properties are available before calculation
        if all(prop in props for prop in ['logp', 'mw', 'tpsa', 'hbd', 'charge']):
             try:
                 bbb_score = sum([
                     1 <= props['logp'] <= 3,
                     props['mw'] <= 400,
                     props['tpsa'] < 90,
                     props['hbd'] <= 3,
                     props['charge'] == 0
                 ])

                 st.metric("BBB Score", f"{bbb_score}/5",
                          "High penetration" if bbb_score >=4 else "Moderate" if bbb_score>=2 else "Low")
             except Exception as e:
                 st.error(f"Could not calculate BBB score: {str(e)}")
        else:
             st.warning("Missing properties for BBB prediction.")


elif app_mode == "🧩 Scaffold Analysis":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">🧩 Scaffold Analysis</h1>
        <p style="color:black; margin:0; opacity:0.8;">Decompose and analyze molecular scaffolds</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input() # Add molecule input section

    if not mol:
        st.warning("No molecule selected")
        st.stop()

    try:
        # Generate and display query scaffold with error handling
        scaffold_mol = None
        try:
            scaffold_mol = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else "N/A"
        except Exception as e:
            st.error(f"Could not generate scaffold for the selected molecule: {str(e)}")
            st.stop()


        if not scaffold_mol:
             st.warning("Could not generate a valid scaffold for the selected molecule.")
             st.stop()

        st.write("### 🏗️ Query Scaffold")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(Draw.MolToImage(scaffold_mol, size=(300,300)),
                     use_container_width=True)
        with col2:
            st.metric("SMILES", f"`{scaffold_smiles}`")
            st.metric("Ring Count", Lipinski.RingCount(scaffold_mol))
            st.metric("Aromatic Rings", Lipinski.NumAromaticRings(scaffold_mol))

        # Database scaffold analysis
        st.write("### 🔍 Similar Scaffolds in Database")

        @st.cache_data(ttl=3600)
        def generate_scaffolds(_data):
            scaffolds = []
            mols = []
            for smiles in _data['smiles']:
                try:
                    m = Chem.MolFromSmiles(str(smiles))
                    if m:
                        s = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(m)
                        scaffolds.append(Chem.MolToSmiles(s) if s else None)
                        mols.append(s) # Store scaffold mol object
                    else:
                        scaffolds.append(None)
                        mols.append(None)
                except Exception as e:
                    st.warning(f"Could not process SMILES {smiles} for scaffold: {str(e)}")
                    scaffolds.append(None)
                    mols.append(None)
            return scaffolds, mols

        with st.spinner("Analyzing database scaffolds..."):
            data['scaffold_smiles'], data['scaffold_mol'] = generate_scaffolds(data)
            valid_data = data[data['scaffold_mol'].notna()].copy().reset_index(drop=True) # Use scaffold_mol for validity check

            if valid_data.empty:
                st.warning("No valid scaffolds found in database")
                st.stop()

            # Calculate scaffold similarities
            try:
                fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                query_fp = fpgen.GetFingerprint(scaffold_mol)

                # Calculate similarities for valid scaffolds
                valid_data['similarity'] = valid_data['scaffold_mol'].apply(
                    lambda s_mol: DataStructs.TanimotoSimilarity(query_fp, fpgen.GetFingerprint(s_mol))
                )

                top_scaffolds = valid_data.nlargest(5, 'similarity')

                # Display results
                for _, row in top_scaffolds.iterrows():
                    with st.expander(f"✨ {row['generic_name']} (Similarity: {row['similarity']:.2f})"):
                        cols = st.columns([1, 3])
                        with cols[0]:
                             if row['scaffold_mol']:
                                st.image(Draw.MolToImage(row['scaffold_mol'],
                                        width=200))
                             else:
                                 st.write("Scaffold image not available")

                        with cols[1]:
                            st.write(f"**Original Compound:** {row['generic_name']}")
                            st.code(f"Scaffold SMILES: {row['scaffold_smiles']}")
                            st.code(f"Full SMILES: {row['smiles']}")
            except Exception as e:
                 st.error(f"Error calculating scaffold similarities: {str(e)}")


    except Exception as e:
        st.error(f"Scaffold analysis failed: {str(e)}")


elif app_mode == "🖥️ Virtual Screening":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">🖥️ Virtual Screening</h1>
        <p style="color:black; margin:0; opacity:0.8;">Screen a list of compounds from an uploaded file against the built-in drug database to find similar molecules</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---") # Use custom HR

    st.write("""
    ### 📄 Upload Compounds for Screening
    Upload a CSV or TXT file containing a list of compounds. The file should have a column named 'SMILES'
    if it's a CSV, or just one SMILES per line if it's a TXT file.
    """)

    # Sample file download
    sample_smiles = "smiles,name\nCCO,Ethanol\nCCN,Ethylamine\nC1=CC=CC=C1,Benzene\nINVALID_SMILES,BadOne\nCC(=O)OC1=CC=CC=C1C(=O)O,Aspirin" # Add an invalid one for demo
    st.download_button(
        "📥 Download Sample SMILES File (CSV)",
        data=sample_smiles,
        file_name="sample_screening_smiles.csv",
        mime="text/csv",
        key="download_sample_screening"
    )

    uploaded_file = st.file_uploader("Choose a file (CSV or TXT)", type=['csv', 'txt'], key="screening_uploader")

    if not uploaded_file:
        st.info("Please upload a file to begin.")
        st.stop()

    try:
        # Process uploaded file
        if uploaded_file.name.endswith('.csv'):
            screen_df = pd.read_csv(uploaded_file)
            if 'smiles' not in screen_df.columns:
                st.error("❌ CSV file must contain a column named 'smiles'.")
                st.stop()
            smiles_col = 'smiles'
        else:  # txt
            screen_df = pd.DataFrame({
                'smiles': [line.decode('utf-8').strip() for line in uploaded_file]
            })
            smiles_col = 'smiles' # The only column name

        st.success(f"✅ Successfully loaded '{uploaded_file.name}'. Found {len(screen_df)} rows.")
        st.dataframe(screen_df.head(), use_container_width=True) # Show a preview

        # Convert SMILES to molecules with robust error handling
        @st.cache_data(show_spinner=False) # Don't show spinner here, use custom status
        def process_screening_molecules(_df, smiles_col):
            processed_mols = []
            processed_smiles = []
            original_indices = []
            invalid_indices = []
            pains_flags = []
            brenk_flags = []
            nih_flags = []
            all_filter_matches = []

            # Initialize filter catalogs
            params_pains = FilterCatalogParams()
            params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            catalog_pains = FilterCatalog(params_pains)
            
            params_brenk = FilterCatalogParams()
            params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            catalog_brenk = FilterCatalog(params_brenk)
            
            params_nih = FilterCatalogParams()
            params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
            catalog_nih = FilterCatalog(params_nih)
            
            params_all = FilterCatalogParams()
            params_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
            catalog_all = FilterCatalog(params_all)

            for i, smi in enumerate(_df[smiles_col]):
                try:
                    mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
                    if mol:
                        Chem.SanitizeMol(mol) # Attempt sanitization
                        processed_mols.append(mol)
                        processed_smiles.append(str(smi))
                        original_indices.append(i)
                        
                        # Check for structural alerts
                        pains_flag = catalog_pains.HasMatch(mol)
                        brenk_flag = catalog_brenk.HasMatch(mol)
                        nih_flag = catalog_nih.HasMatch(mol)
                        
                        pains_flags.append(pains_flag)
                        brenk_flags.append(brenk_flag)
                        nih_flags.append(nih_flag)
                        
                        # Get all filter matches details
                        all_matches = catalog_all.GetMatches(mol)
                        if all_matches:
                            filter_details = "; ".join([
                                f"{entry.GetProp('FilterSet')}: {entry.GetDescription()}"
                                for entry in all_matches
                            ])
                            all_filter_matches.append(filter_details)
                        else:
                            all_filter_matches.append("None")
                    else:
                        invalid_indices.append(i)
                        pains_flags.append(False)
                        brenk_flags.append(False)
                        nih_flags.append(False)
                        all_filter_matches.append("Invalid SMILES")
                except Exception as e:
                    invalid_indices.append(i)
                    pains_flags.append(False)
                    brenk_flags.append(False)
                    nih_flags.append(False)
                    all_filter_matches.append("Processing Error")
                    continue
            
            return (processed_mols, processed_smiles, original_indices, invalid_indices,
                   pains_flags, brenk_flags, nih_flags, all_filter_matches)

        with st.status("Processing molecules from file...", expanded=True) as status_mol_process:
            (screen_mols, screen_smiles_valid, original_indices, invalid_indices,
             pains_flags, brenk_flags, nih_flags, all_filter_matches) = process_screening_molecules(screen_df, smiles_col)

            if not screen_mols:
                status_mol_process.update(label="❌ No valid molecules found in the uploaded file.", state="error")
                st.error("No valid molecules could be processed from the uploaded file. Please check the SMILES strings.")
                st.stop()

            status_mol_process.update(label=f"✅ Successfully processed {len(screen_mols)} molecules. {len(invalid_indices)} invalid/skipped.", state="complete")

        # Display structural alert information
        st.markdown("---")
        st.write("### 🚨 Structural Alert Screening")

# First check if we have all required data
def run_structural_alert_screening():
    try:
        # Verify all required variables exist
        required_vars = ['screen_smiles_valid', 'pains_flags', 'brenk_flags', 'nih_flags', 'all_filter_matches']
        missing_vars = [var for var in required_vars if var not in globals()]
        if missing_vars:
            raise NameError(f"Missing required variables: {', '.join(missing_vars)}")

        # Validate array lengths
        base_length = len(screen_smiles_valid)
        if not all(len(globals()[var]) == base_length for var in required_vars[1:]):
            raise ValueError(f"All arrays must have same length as screen_smiles_valid ({base_length})")

        # Create DataFrame
        alert_data = {
            'SMILES': screen_smiles_valid,
            'PAINS Alert': pains_flags,
            'Brenk Alert': brenk_flags,
            'NIH Alert': nih_flags,
            'All Filter Matches': all_filter_matches
        }

        # Add names if available
        if 'name' in screen_df.columns and len(screen_df) == base_length:
            alert_data['Name'] = screen_df['name'].values

        alert_df = pd.DataFrame(alert_data)

        # Display results
        st.dataframe(
            alert_df,
            use_container_width=True,
            column_config={
                "PAINS Alert": st.column_config.CheckboxColumn("PAINS Alert"),
                "Brenk Alert": st.column_config.CheckboxColumn("Brenk Alert"), 
                "NIH Alert": st.column_config.CheckboxColumn("NIH Alert"),
                "All Filter Matches": "Matched Filters"
            }
        )

        # Show summary metrics
        cols = st.columns(3)
        cols[0].metric("PAINS Alerts", f"{sum(pains_flags)}/{len(pains_flags)}")
        cols[1].metric("Brenk Alerts", f"{sum(brenk_flags)}/{len(brenk_flags)}")
        cols[2].metric("NIH Alerts", f"{sum(nih_flags)}/{len(nih_flags)}")

        # Download button
        st.download_button(
            "💾 Download Results",
            alert_df.to_csv(index=False).encode('utf-8'),
            "structural_alerts.csv",
            "text/csv"
        )

    except NameError as e:
        st.error(f"Configuration error: {str(e)}")
    except ValueError as e:
        st.error(f"Data error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

# Execute the screening function
run_structural_alert_screening()
        
        st.markdown("---") # Use custom HR
        # Screening parameters
        st.write("### ⚙️ Screening Parameters")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                fp_type_screen = st.selectbox("Fingerprint Type", ["Morgan", "MACCS", "RDKit"], key="fp_type_screen")
            with col2:
                threshold_screen = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, step=0.05, key="threshold_screen", help="Minimum Tanimoto similarity score to consider a match.")
            with col3:
                max_matches_screen = st.slider("Max Matches per Query", 1, 50, 10, step=1, key="max_matches_screen", help="Maximum number of similar compounds to retrieve from the database for each query molecule.")

        st.markdown("---") # Use custom HR

        if st.button("🚀 Start Virtual Screening", key="start_screening", type="primary", use_container_width=True):
            # Fingerprint calculation with error handling (re-defined for scope)
            def get_fingerprint_screen(m, fp_type):
                try:
                    if m is None: return None
                    if fp_type == "Morgan":
                        return AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
                    elif fp_type == "MACCS":
                        return AllChem.GetMACCSKeysFingerprint(m)
                    elif fp_type == "RDKit":
                        return Chem.RDKFingerprint(m)
                    return None
                except:
                    return None # Return None on failure

            # Precompute database fingerprints with progress
            with st.status(f"Preparing database fingerprints ({fp_type_screen})...", expanded=False) as status_db_fp:
                db_fps = []
                valid_db_indices = []
                for i, row in data.iterrows():
                    db_mol = Chem.MolFromSmiles(str(row['smiles']))
                    fp = get_fingerprint_screen(db_mol, fp_type_screen)
                    if fp is not None:
                        db_fps.append(fp)
                        valid_db_indices.append(i) # Store index of valid database entry

                if not db_fps:
                    status_db_fp.update(label="❌ Could not generate fingerprints for database compounds.", state="error")
                    st.error("Failed to generate fingerprints for the database compounds. Screening cannot proceed.")
                    st.stop()

                status_db_fp.update(label=f"✅ Prepared fingerprints for {len(db_fps)} database compounds.", state="complete")

            # Perform screening
            results = []
            total_queries = len(screen_mols)
            st.write(f"Screening {total_queries} query molecules against {len(db_fps)} database compounds...")
            progress_bar_screening = st.progress(0, text="Screening progress: 0%")

            db_data_valid = data.iloc[valid_db_indices].reset_index(drop=True) # Use only database entries with valid fingerprints

            for i, query_mol in enumerate(screen_mols):
                query_fp = get_fingerprint_screen(query_mol, fp_type_screen)
                if query_fp is None:
                    continue # Skip if query fingerprint fails

                # Calculate similarities against the valid database fingerprints
                similarities = []
                for db_fp in db_fps:
                    try:
                        similarity = DataStructs.TanimotoSimilarity(query_fp, db_fp)
                        similarities.append(similarity)
                    except Exception as e:
                        similarities.append(0.0)

                # Add similarities to the valid database data
                db_data_valid['similarity'] = similarities

                # Find matches above threshold
                matches = db_data_valid[db_data_valid['similarity'] >= threshold_screen]\
                    .sort_values('similarity', ascending=False)\
                    .head(max_matches_screen)

                # Add matches to results list
                original_query_smiles = screen_smiles_valid[i] # Get original SMILES from the valid list
                original_row_index = original_indices[i] # Get original row index
                query_name = screen_df.iloc[original_row_index].get('name', 'N/A') # Try to get 'name' if it exists
                query_pains = pains_flags[i]
                query_brenk = brenk_flags[i]
                query_nih = nih_flags[i]

                for match in matches.itertuples():
                    results.append({
                        'Query SMILES': original_query_smiles,
                        'Query Name': query_name,
                        'Query PAINS Alert': query_pains,
                        'Query Brenk Alert': query_brenk,
                        'Query NIH Alert': query_nih,
                        'Match Name': match.generic_name,
                        'Match SMILES': match.smiles,
                        'Similarity': match.similarity,
                        'Match MW': match.mw,
                        'Match LogP': match.logp
                    })

                # Update progress bar
                progress_bar_screening.progress((i + 1) / total_queries, text=f"Screening progress: {int((i+1)/total_queries*100)}%")

            progress_bar_screening.empty() # Hide progress bar when done

            if results:
                results_df = pd.DataFrame(results)
                st.balloons()
                st.success(f"🎉 Screening complete! Found {len(results_df)} matches above similarity threshold {threshold_screen:.2f}.")

                # Display results table
                st.write("### Screening Results")
                st.dataframe(
                    results_df.sort_values(['Query SMILES', 'Similarity'], ascending=[True, False]),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Similarity": st.column_config.ProgressColumn(
                            "Similarity Score",
                            help="Tanimoto similarity score",
                            format="%.3f",
                            min_value=0,
                            max_value=1
                        ),
                        "Query SMILES": "Query SMILES",
                        "Query Name": "Query Name",
                        "Query PAINS Alert": st.column_config.CheckboxColumn("Query PAINS Alert"),
                        "Query Brenk Alert": st.column_config.CheckboxColumn("Query Brenk Alert"),
                        "Query NIH Alert": st.column_config.CheckboxColumn("Query NIH Alert"),
                        "Match Name": "Database Match Name",
                        "Match SMILES": "Database Match SMILES",
                        "Match MW": st.column_config.NumberColumn("Match MW", format="%.2f"),
                        "Match LogP": st.column_config.NumberColumn("Match LogP", format="%.2f")
                    }
                )

                # Download results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Screening Results (CSV)",
                    csv,
                    "virtual_screening_results.csv",
                    "text/csv",
                    key="download_screening_results"
                )
            else:
                st.warning(f"No matches found above the similarity threshold {threshold_screen:.2f} for any query molecule.")

    except Exception as e:
        st.error(f"An unexpected error occurred during virtual screening: {e}")
        st.write(traceback.format_exc())



elif app_mode == "⚗️ Compound Optimization":
    st.markdown("""
    <div class="header">
        <h1 style="color:black; margin:0;">⚗️ Compound Optimization</h1>
        <p style="color:black; margin:0; opacity:0.8;">Get suggestions for improving drug properties</p>
    </div>
    """, unsafe_allow_html=True)

    display_molecule_input() # Add molecule input section

    if not mol:
        st.warning("No molecule selected")
        st.stop()

    # Display molecule
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(Draw.MolToImage(mol, size=(300, 300)), use_container_width=True)

    # Calculate properties with error handling
    props = {}
    try:
        props = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'Rings': Lipinski.RingCount(mol)
        }
    except Exception as e:
         st.error(f"Error calculating properties for optimization: {str(e)}")
         st.stop()


    # Lipinski evaluation
    lipinski = {
        'MW <500': props.get('MW', float('inf')) < 500,
        'LogP <5': props.get('LogP', float('inf')) < 5,
        'HBD ≤5': props.get('HBD', float('-inf')) <= 5,
        'HBA ≤10': props.get('HBA', float('-inf')) <= 10
    }

    with col2:
        # Property cards
        st.metric("Molecular Weight", f"{props.get('MW', np.nan):.1f}" if 'MW' in props else "N/A")
        st.metric("LogP", f"{props.get('LogP', np.nan):.2f}" if 'LogP' in props else "N/A")
        st.metric("Lipinski Compliance",
                f"{sum(lipinski.values())}/4",
                help="\n".join(f"{k}: {'✔' if v else '✖'}" for k,v in lipinski.items()))

    # Toxicity alerts
    toxic_alerts = {
        "Nitroso": "[N](=O)",
        "Michael Acceptor": "C=CC=O",
        "Aniline": "c1ccc(cc1)N"
    }
    toxic_found = []
    for name, smarts in toxic_alerts.items():
        try:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                toxic_found.append(name)
        except Exception as e:
            st.warning(f"Could not check toxicophore {name}: {str(e)}")


    if toxic_found:
        st.error(f"⚠️ Potential Toxicophores: {', '.join(toxic_found)}")
    else:
        st.success("✅ No common toxicophores detected")

    # Bioisosteres
    bioisosteres = {
        "[C](=O)[O-]": ["SO₃H", "PO₃H₂", "Tetrazole"], # Carboxylic acid
        "c1ccccc1": ["Pyridyl", "Thiophene", "Imidazole"], # Phenyl
        "[OH]": ["NH₂", "F", "CONH₂"], # Hydroxyl
        "[NH2]": ["[OH]", "CH₃"] # Amine
    }

    st.markdown("### 🔄 Bioisostere Suggestions")
    st.caption("Suggested replacements for common functional groups based on bioisosterism principles.")

    found_bioisosteres = False
    for smarts, replacements in bioisosteres.items():
        try:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                st.markdown(f"<div class='card'><b>{smarts}</b> group found. Consider replacing with: {', '.join(replacements)}</div>", unsafe_allow_html=True)
                found_bioisosteres = True
        except Exception as e:
            st.warning(f"Could not check for bioisostere {smarts}: {str(e)}")

    if not found_bioisosteres:
        st.info("No common bioisosteric groups found in the molecule.")


    # Optimization tips
    st.markdown("### 🧠 Optimization Tips")
    st.caption("General strategies for improving drug-like properties.")
    st.markdown("""
    <div class="card">
    <ul>
        <li><b>Solubility</b>: Increase polarity (e.g., add OH, NH, COOH), reduce LogP, reduce MW.</li>
        <li><b>Permeability</b>: Reduce TPSA, reduce HBD, maintain LogP in optimal range (1-3).</li>
        <li><b>Metabolic Stability</b>: Replace metabolically labile groups (e.g., esters, amides, ethers) with more stable bioisosteres. Consider deuteration.</li>
        <li><b>Synthetic Ease</b>: Aim for simpler structures (e.g., fewer chiral centers, fewer complex rings), keep heavy atom count reasonable (<~35).</li>
        <li><b>Toxicity</b>: Remove toxicophores, modify reactive functional groups.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# Add clear cache button in foote
   

elif app_mode == "ℹ️ About":
    st.markdown("""
        <style>
            .about-header {
                color: #004080;
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 20px;
                text-align: center;
            }
            .about-section {
                background-color: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .profile-img {
                border-radius: 50%;
                width: 150px;
                height: 150px;
                object-fit: cover;
                border: 4px solid #004080;
                display: block;
            }
            .section-title {
                color: #004080;
                font-size: 1.5rem;
                font-weight: bold;
                margin-bottom: 15px;
                border-bottom: 2px solid #004080;
                padding-bottom: 8px;
            }
            .contact-info {
                background-color: #e9f7fe;
                padding: 15px;
                border-radius: 10px;
                margin-top: 10px;
            }
            .profile-container {
                display: flex;
                align-items: center;
                gap: 30px;
                margin-bottom: 20px;
            }
            .profile-text {
                flex: 1;
            }
            .github-btn {
                background-color: #333;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                text-decoration: none;
                display: inline-block;
                margin-top: 10px;
                font-weight: bold;
            }
            .github-btn:hover {
                background-color: #555;
            }
            @media (max-width: 768px) {
                .profile-container {
                    flex-direction: column;
                    text-align: center;
                }
                .profile-img {
                    margin: 0 auto;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="about-header">ℹ️ About</h1>', unsafe_allow_html=True)

    # Author Section
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<img src="https://media.licdn.com/dms/image/v2/D5603AQGH9FTKrbtsBQ/profile-displayphoto-shrink_200_200/B56ZavaEF2GgAc-/0/1746699569459?e=1752105600&v=beta&t=Ka2PPXH6ii9rMbIQ3WPfNO2meHi3T-Qb03xAfME1fZs" class="profile-img" alt="Ankita Chavan">', unsafe_allow_html=True)
    st.markdown('''
        <div class="profile-text">
            <h2 class="section-title">👩‍🔬 About the Author</h2>
            <p style="font-size: 1.1rem;"><strong>Ankita Chavan</strong></p>
            <p>Currently pursuing a Master's degree in Bioinformatics from DES (Deccan Education Society) Pune University, Pune.</p>
            <p>Passionate about computational drug discovery, cheminformatics, and developing bioinformatics tools to solve biological problems.</p>
            <p>This web application is part of my academic project under the guidance of Dr. Kushagra Kashyap.</p>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Web Server Information
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🌐 About This Web Server</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p>This web server provides advanced cheminformatics tools for:</p>
        <ul>
            <li>Molecular similarity searching using fingerprint techniques</li>
            <li>Compound database screening</li>
            <li>Molecular property calculation and visualization</li>
            <li>Drug discovery research support</li>
        </ul>
        <p>The application is built using:</p>
        <ul>
            <li>Python with RDKit for cheminformatics</li>
            <li>Streamlit for web interface</li>
            <li>Pandas for data handling</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Mentor Section
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<img src="https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/profile-displayphoto-shrink_400_400/B56ZZI.WrdH0Ag-/0/1744981029051?e=1752105600&v=beta&t=F4QBDSEgjUvnBS00xPkKqPTLI0jQaMpYefaOzARY1Yg" class="profile-img" alt="Dr. Kushagra Kashyap">', unsafe_allow_html=True)
    st.markdown('''
        <div class="profile-text">
            <h2 class="section-title">👨‍🏫 About the Mentor</h2>
            <p style="font-size: 1.1rem;"><strong>Dr. Kushagra Kashyap</strong></p>
            <p>Assistant Professor at DES (Deccan Education Society) Pune University.</p>
            <p>Specializes in Bioinformatics and Cheminformatics, with research interests in computational drug discovery and molecular modeling.</p>
            <p>Provides guidance on bridging the gap between biological sciences and computational technologies.</p>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Contact Information
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📬 Contact & Feedback</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p>For any questions, suggestions, or feedback about this web application:</p>
        <div class="contact-info">
            <p><strong>📧 Email:</strong> 3522411012@despu.edu.in</p>
            <p><strong>🔗 LinkedIn:</strong> <a href="https://www.linkedin.com/in/ankita-chavan-408709226" target="_blank">Ankita Chavan's Profile</a></p>
            <p><strong>💻 Source Code:</strong> 
                <a href="https://github.com/AnkitaSchavan/DrugD/edit/main/p1.py" target="_blank" class="github-btn">
                    View on GitHub
                </a>
            </p>
        </div>
        <p style="margin-top: 15px; font-style: italic;">Your feedback helps improve this tool for the research community!</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
if st.button("Clear All Cache", help="Reset all cached data", key='clear_cache_button'):
    st.cache_data.clear()
    st.success("Cache cleared successfully!")
