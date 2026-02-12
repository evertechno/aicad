import streamlit as st
import requests
import pandas as pd
import trimesh
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from stl_viewer import stl_viewer # Note: streamlit-3d-viewer provides stl_viewer, which works for OBJ too

# --- Configuration and Page Setup ---
st.set_page_config(
    layout="wide",
    page_title="AI 3D Design Studio",
    page_icon="🤖"
)

st.title("🤖 AI 3D Design Studio")
st.markdown("""
Welcome to the future of 3D modeling. This tool uses a powerful AI to transform your text prompts into detailed 3D mesh codes, ready for rendering and manufacturing.
""")

# --- AI Core Functions (2-Tier Structure) ---

# Check for secrets
try:
    ACCOUNT_ID = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
    AUTH_TOKEN = st.secrets["CLOUDFLARE_AUTH_TOKEN"]
except KeyError:
    st.error("Cloudflare credentials not found in st.secrets. Please add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN to your .streamlit/secrets.toml file.")
    st.stop()

# System prompts for our 2-tier AI structure
ENHANCER_SYSTEM_PROMPT = """
You are an expert CAD designer and engineering assistant. Your task is to take a user's design idea and expand it into a highly detailed, unambiguous technical specification. This specification will be fed to another AI that generates 3D mesh code. Be extremely precise about dimensions (in mm), geometric properties, topology, key features, symmetries, and the target format (Wavefront OBJ). The output must be a clear, step-by-step instruction set for the modeling AI. Assume the model will be centered at the origin (0,0,0). Break down complex shapes into primitive components and describe how they connect.
"""

GENERATOR_SYSTEM_PROMPT = """
You are a master 3D modeling AI. Your only purpose is to generate valid, clean, and production-ready Wavefront OBJ (.obj) mesh code based on the provided technical specification. 
**CRITICAL INSTRUCTIONS:**
1.  **ONLY an .obj file format must be returned.**
2.  **DO NOT** output any text, explanation, conversation, or markdown formatting (like ```obj) before or after the code.
3.  Your entire response must be only the raw text of the .obj file, starting with vertex definitions (`v`), vertex normals (`vn`), and faces (`f`).
4.  Ensure the mesh is watertight and has correct face normals for 3D printing.
5.  The model must be centered at the origin (0,0,0).
"""

@st.cache_data(ttl=3600, show_spinner=False)
def run_cloudflare_ai(system_prompt, user_prompt):
    """Generic function to call the Cloudflare AI API."""
    try:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 16000 
            }
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success") and result.get("result", {}).get("response"):
            return result["result"]["response"].strip()
        else:
            return f"Error: API call failed. Response: {result}"
    except requests.exceptions.RequestException as e:
        return f"Error: Network or API request failed. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def generate_mesh_code(user_prompt: str):
    """Implements the 2-tier generation process."""
    # Tier 1: Enhance the prompt
    st.info("Tier 1: Enhancing prompt for technical specificity...")
    enhanced_prompt = run_cloudflare_ai(ENHANCER_SYSTEM_PROMPT, user_prompt)
    if enhanced_prompt.startswith("Error:"):
        st.error(f"Failed at Tier 1 (Prompt Enhancer): {enhanced_prompt}")
        return None, None
    
    st.session_state.enhanced_prompt = enhanced_prompt

    # Tier 2: Generate the mesh code from the enhanced prompt
    st.info("Tier 2: Generating production-ready OBJ mesh code...")
    mesh_code = run_cloudflare_ai(GENERATOR_SYSTEM_PROMPT, enhanced_prompt)
    if mesh_code.startswith("Error:"):
        st.error(f"Failed at Tier 2 (Mesh Generator): {mesh_code}")
        return enhanced_prompt, None

    # Clean the output to ensure it's valid OBJ
    lines = [line for line in mesh_code.split('\n') if line.strip().startswith(('v ', 'f ', 'vn ', '#', 'o ', 'g '))]
    cleaned_mesh_code = "\n".join(lines)
    
    st.session_state.generated_mesh_code = cleaned_mesh_code
    return enhanced_prompt, cleaned_mesh_code

def render_mesh(obj_code, key="viewer"):
    """Renders mesh code using trimesh and stl_viewer."""
    if not obj_code:
        st.warning("No mesh code to render.")
        return

    try:
        # Use trimesh to load the OBJ data from a string
        # We wrap the string in a file-like object
        file_obj = io.StringIO(obj_code)
        mesh = trimesh.load(file_obj, file_type='obj')

        # Use stl_viewer to render the mesh
        # It requires vertices and faces as separate lists of lists/tuples
        with st.container(border=True):
            st.subheader("3D Model Viewer")
            stl_viewer(
                model_path=None, 
                points=mesh.vertices.tolist(),
                faces=mesh.faces.tolist(),
                key=key,
                height=500
            )
    except Exception as e:
        st.error(f"Failed to render 3D model. The generated code might be invalid. Error: {e}")
        st.code(obj_code, language='text')

# --- Streamlit Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🎨 Create Mesh Codes", 
    "🛠️ Render Models & Design", 
    "🤖 AI & Automations"
])

# --- TAB 1: Create Mesh Codes ---
with tab1:
    st.header("Generate 3D Models from Text")
    st.info("Describe the object you want to create. Be as descriptive as you like. The AI will first refine your idea into a technical spec, then generate the 3D mesh code.")
    
    user_prompt = st.text_area("Enter your design idea here:", 
                               height=150, 
                               placeholder="e.g., A modern, ergonomic computer mouse with two buttons and a scroll wheel, 120mm long, 65mm wide.",
                               key="main_prompt")

    if st.button("✨ Generate 3D Model", type="primary", use_container_width=True):
        if user_prompt:
            st.session_state.enhanced_prompt = ""
            st.session_state.generated_mesh_code = ""
            with st.spinner("AI is thinking... This may take a minute."):
                enhanced_prompt, mesh_code = generate_mesh_code(user_prompt)
            
            if mesh_code:
                st.success("Successfully generated 3D model!")
        else:
            st.warning("Please enter a design idea.")

    if 'generated_mesh_code' in st.session_state and st.session_state.generated_mesh_code:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generated Mesh Code (.obj)")
            st.code(st.session_state.generated_mesh_code, language='text', height=500)
            st.download_button(
                label="📥 Download .obj file",
                data=st.session_state.generated_mesh_code,
                file_name=f"{user_prompt[:20].replace(' ','_')}.obj",
                mime="text/plain"
            )
            
            with st.expander("View Enhanced Technical Prompt (Tier 1 Output)"):
                st.markdown(st.session_state.enhanced_prompt)

        with col2:
            render_mesh(st.session_state.generated_mesh_code, key="tab1_viewer")

# --- TAB 2: Render Models & Design ---
with tab2:
    st.header("Import, Render, and Export 3D Designs")
    st.info("Upload a 3D model file (OBJ, STL, GLB, etc.) to view and convert it to other formats.")

    uploaded_file = st.file_uploader(
        "Choose a 3D model file", 
        type=['obj', 'stl', 'ply', 'glb', 'gltf'],
        key="file_uploader"
    )

    if 'mesh' not in st.session_state:
        st.session_state.mesh = None

    if uploaded_file is not None:
        try:
            # trimesh can load from a file-like object directly
            st.session_state.mesh = trimesh.load(uploaded_file, file_type=uploaded_file.name.split('.')[-1])
            st.success(f"Successfully loaded `{uploaded_file.name}`.")
        except Exception as e:
            st.error(f"Failed to load file. Error: {e}")
            st.session_state.mesh = None

    if st.session_state.mesh is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("3D Model Viewer")
            with st.container(border=True):
                 stl_viewer(
                    model_path=None, 
                    points=st.session_state.mesh.vertices.tolist(),
                    faces=st.session_state.mesh.faces.tolist(),
                    key="tab2_viewer",
                    height=600
                )
        
        with col2:
            st.subheader("Model Information")
            st.info(f"""
            - **Vertices:** {len(st.session_state.mesh.vertices)}
            - **Faces:** {len(st.session_state.mesh.faces)}
            - **Is Watertight?** {st.session_state.mesh.is_watertight}
            """)

            st.subheader("Export Model")
            export_format = st.selectbox("Select export format:", ["STL", "OBJ", "GLB"])

            if st.button(f"Export as {export_format}", use_container_width=True):
                with st.spinner(f"Converting to {export_format}..."):
                    try:
                        # Export to an in-memory buffer
                        buffer = io.BytesIO()
                        st.session_state.mesh.export(buffer, file_type=export_format.lower())
                        buffer.seek(0)

                        st.download_button(
                            label=f"📥 Download .{export_format.lower()} file",
                            data=buffer,
                            file_name=f"exported_model.{export_format.lower()}",
                            mime=f"model/{export_format.lower()}",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Failed to export. Error: {e}")

# --- TAB 3: AI & Automations ---
with tab3:
    st.header("Batch Generate Designs with AI")
    st.info("""
    Upload a CSV file containing a column named **"design ideas"**. The AI will process each idea in parallel to generate a 3D model.
    
    **Example CSV format:**
    ```
    design ideas,category
    A simple coffee mug,kitchenware
    A hexagonal bolt M10,hardware
    A minimalist phone stand,accessories
    ```
    """)

    uploaded_csv = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            if "design ideas" not in df.columns:
                st.error("CSV must contain a column named 'design ideas'.")
            else:
                st.success(f"Found {len(df)} design ideas in the CSV.")
                st.dataframe(df)

                if st.button("🚀 Start Batch Generation", type="primary", use_container_width=True):
                    # The worker function for the thread pool
                    def process_idea(idea):
                        _, mesh_code = generate_mesh_code(idea)
                        return idea, mesh_code

                    ideas = df["design ideas"].dropna().tolist()
                    results = {}
                    
                    progress_bar = st.progress(0, text="Starting batch generation...")
                    
                    with ThreadPoolExecutor(max_workers=5) as executor: # Use up to 5 parallel threads
                        future_to_idea = {executor.submit(process_idea, idea): idea for idea in ideas}
                        
                        total_tasks = len(ideas)
                        completed_tasks = 0

                        for future in as_completed(future_to_idea):
                            idea = future_to_idea[future]
                            try:
                                _, mesh_code = future.result()
                                results[idea] = mesh_code
                            except Exception as exc:
                                results[idea] = f"Error generating model: {exc}"
                            
                            completed_tasks += 1
                            progress_val = completed_tasks / total_tasks
                            progress_bar.progress(progress_val, text=f"Processing: {idea[:40]}...")

                    progress_bar.progress(1.0, text="Batch generation complete!")
                    st.session_state.batch_results = results
                    
        except Exception as e:
            st.error(f"Failed to process CSV file. Error: {e}")

    if 'batch_results' in st.session_state:
        st.subheader("Batch Generation Results")
        for idea, mesh_code in st.session_state.batch_results.items():
            with st.expander(f"**Idea:** {idea}"):
                if mesh_code and not mesh_code.startswith("Error"):
                    st.code(mesh_code, language="text", height=300)
                    st.download_button(
                        label="📥 Download .obj",
                        data=mesh_code,
                        file_name=f"{idea[:20].replace(' ','_')}.obj",
                        mime="text/plain",
                        key=f"download_{idea}"
                    )
                else:
                    st.error(f"Could not generate model for this idea. Reason: {mesh_code}")
