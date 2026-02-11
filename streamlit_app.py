import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="AI 3D Mesh Designer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
    }
    .mesh-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: #f8f9fa;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_meshes' not in st.session_state:
    st.session_state.generated_meshes = []
if 'mesh_history' not in st.session_state:
    st.session_state.mesh_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# API Configuration
def get_cloudflare_api_config():
    """Get Cloudflare API configuration from secrets"""
    try:
        account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
        auth_token = st.secrets["CLOUDFLARE_AUTH_TOKEN"]
        return account_id, auth_token
    except Exception as e:
        st.error(f"Error loading API credentials: {e}")
        return None, None

def call_ai_api(messages, max_tokens=16000):
    """Call Cloudflare AI API with enhanced error handling"""
    account_id, auth_token = get_cloudflare_api_config()
    
    if not account_id or not auth_token:
        return None
    
    try:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "messages": messages,
                "max_tokens": max_tokens
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result['result']['response']
            else:
                st.error(f"API Error: {result.get('errors', 'Unknown error')}")
                return None
        else:
            st.error(f"HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def enhance_prompt(user_prompt):
    """Tier 1: Enhance user prompt with detailed specifications"""
    system_message = """You are an expert CAD engineer and industrial designer. Transform basic design ideas into COMPLETE, MANUFACTURABLE specifications.

For EVERY design, include:

1. OVERALL DIMENSIONS (exact measurements in mm)
2. FUNCTIONAL FEATURES (handles, grips, openings, connections)
3. STRUCTURAL DETAILS (wall thickness, ribs, supports)
4. ERGONOMIC CONSIDERATIONS (grip zones, comfortable angles)
5. AESTHETIC ELEMENTS (curves, chamfers, fillets, textures)
6. ASSEMBLY POINTS (if multi-part)
7. MATERIAL CONSIDERATIONS (draft angles for molding, etc.)

CRITICAL: Think like a product designer creating something REAL that will be manufactured. Include:
- All curves and fillets (specify radius)
- Draft angles for manufacturing
- Grip textures or patterns where hands touch
- Functional details (drainage holes, ventilation, cable management)
- Proportions that look professional
- Weight distribution considerations

Example transformation:
"Create a coffee mug" → 
"Design a 350ml ceramic coffee mug with these specs:
- Main body: Cylindrical, 95mm tall, 80mm outer diameter, 3mm wall thickness
- Taper: 2-degree inward slope from rim to base for structural stability
- Handle: D-shaped cross-section (12mm x 15mm), positioned at 45° angle, ergonomic curve with 25mm radius at top, 30mm radius at bottom
- Rim: 2mm rounded edge for comfort
- Base: Flat circular base 75mm diameter with 1mm recess for stability
- Interior: Smooth glazed surface, slight concave bottom (3mm radius)
- Exterior: Subtle vertical grooves (1mm deep, 5mm spacing) for grip
- Drainage: Small 2mm hole at handle junction to prevent water pooling"

Make EVERY design this detailed. Think manufacturing, ergonomics, aesthetics."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Transform this into a complete, manufacturable specification:\n\n{user_prompt}\n\nBe extremely detailed with measurements, features, and functional elements."}
    ]
    
    return call_ai_api(messages, max_tokens=6000)

def generate_mesh_code(enhanced_prompt):
    """Tier 2: Generate production-ready mesh code from enhanced specifications"""
    system_message = """You are a precision mesh generator creating REAL, MANUFACTURABLE 3D models.

OUTPUT REQUIREMENTS:
1. Generate COMPLETE, DETAILED meshes with sufficient vertices for smooth curves
2. For curved surfaces: Use AT LEAST 16-32 vertices per circle/curve
3. Include ALL functional features (handles, grips, holes, textures)
4. Create proper topology (quads where possible, clean edge loops)
5. Add geometric details like chamfers, fillets, grooves, patterns
6. Make it look PROFESSIONAL, not like a basic primitive

MESH DENSITY GUIDELINES:
- Simple objects: 200-500 vertices minimum
- Objects with curves: 500-2000 vertices
- Complex objects: 2000-5000 vertices
- Add detail where it matters (edges, functional areas)

JSON FORMAT:
{
  "mesh_name": "professional_descriptive_name",
  "vertices": [
    [x1, y1, z1], [x2, y2, z2], ... // Use floating point, position carefully
  ],
  "faces": [
    [v1, v2, v3], [v4, v5, v6], ... // Triangle faces, counter-clockwise winding
  ],
  "normals": [
    [nx1, ny1, nz1], ... // Unit vectors, properly calculated
  ],
  "metadata": {
    "units": "millimeters",
    "scale": 1.0,
    "origin": [0, 0, 0],
    "bounds": {"min": [x, y, z], "max": [x, y, z]},
    "vertex_count": N,
    "face_count": M,
    "description": "Brief description of what this is"
  },
  "materials": {
    "color": [r, g, b],  // Realistic colors (0-1 range)
    "roughness": 0.3,    // Material finish
    "metallic": 0.0
  }
}

CRITICAL RULES:
1. Generate REAL mesh coordinates - calculate actual vertex positions
2. For cylinders: Create circular cross-sections with proper trigonometry
3. For curves: Use bezier/spline approximations with enough segments
4. Include fine details: chamfers use 4-8 vertices, rounded corners need proper arc subdivision
5. Ensure watertight mesh: All edges shared by exactly 2 faces
6. Center the model at origin, ground plane at Z=0 if applicable

EXAMPLES OF GOOD VS BAD:

BAD (Generic cube):
- 8 vertices only
- No details
- Looks like a test primitive

GOOD (Detailed box with features):
- 200+ vertices
- Rounded corners with proper 8-vertex fillets
- Chamfered edges
- Textured surfaces (small geometric details)
- Functional features (latches, hinges if applicable)

Generate ONLY the JSON. Make it production-ready. This will be used for REAL manufacturing."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Generate a highly detailed, production-ready 3D mesh for:\n\n{enhanced_prompt}\n\nCreate a COMPLETE mesh with all features, curves, and details described. Use enough vertices to make curves smooth and details visible. Return ONLY valid JSON."}
    ]
    
    return call_ai_api(messages, max_tokens=16000)

def parse_mesh_json(mesh_code_str):
    """Parse and validate mesh JSON"""
    try:
        # Try to extract JSON from markdown code blocks
        if "```json" in mesh_code_str:
            start = mesh_code_str.find("```json") + 7
            end = mesh_code_str.find("```", start)
            mesh_code_str = mesh_code_str[start:end].strip()
        elif "```" in mesh_code_str:
            start = mesh_code_str.find("```") + 3
            end = mesh_code_str.find("```", start)
            mesh_code_str = mesh_code_str[start:end].strip()
        
        mesh_data = json.loads(mesh_code_str)
        
        # Quality validation
        vertex_count = len(mesh_data.get('vertices', []))
        face_count = len(mesh_data.get('faces', []))
        
        warnings = []
        
        if vertex_count < 50:
            warnings.append(f"⚠️ Low detail mesh: Only {vertex_count} vertices. Professional meshes typically have 200+ vertices.")
        
        if face_count < 50:
            warnings.append(f"⚠️ Low polygon count: Only {face_count} faces. Consider requesting more detail.")
        
        if vertex_count < 12:
            warnings.append("❌ CRITICAL: Mesh is too simple (basic primitive). Regenerate with 'add more detail and vertices' in prompt.")
        
        if warnings:
            st.warning("Mesh Quality Issues:\n" + "\n".join(warnings))
        else:
            st.success(f"✅ Good quality mesh: {vertex_count} vertices, {face_count} faces")
        
        return mesh_data
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse mesh JSON: {e}")
        return None

def render_mesh_plotly(mesh_data):
    """Render 3D mesh using Plotly"""
    try:
        vertices = np.array(mesh_data.get('vertices', []))
        faces = np.array(mesh_data.get('faces', []))
        
        if len(vertices) == 0 or len(faces) == 0:
            st.warning("No valid mesh data to render")
            return None
        
        # Extract coordinates
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        # Get color from materials
        color = mesh_data.get('materials', {}).get('color', [0.5, 0.7, 1.0])
        color_str = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
        
        # Create mesh3d trace
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=color_str,
                opacity=0.8,
                flatshading=True,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.3
                ),
                lightposition=dict(x=100, y=100, z=100)
            )
        ])
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', backgroundcolor="rgb(240, 240, 240)"),
                yaxis=dict(title='Y', backgroundcolor="rgb(240, 240, 240)"),
                zaxis=dict(title='Z', backgroundcolor="rgb(240, 240, 240)"),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title=dict(
                text=mesh_data.get('mesh_name', 'Generated Mesh'),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    except Exception as e:
        st.error(f"Rendering error: {e}")
        return None

def export_mesh(mesh_data, format='obj'):
    """Export mesh to various formats"""
    try:
        vertices = mesh_data.get('vertices', [])
        faces = mesh_data.get('faces', [])
        mesh_name = mesh_data.get('mesh_name', 'mesh')
        
        if format == 'obj':
            # OBJ format
            obj_content = f"# Generated by AI 3D Mesh Designer\n"
            obj_content += f"o {mesh_name}\n\n"
            
            # Write vertices
            for v in vertices:
                obj_content += f"v {v[0]} {v[1]} {v[2]}\n"
            
            obj_content += "\n"
            
            # Write faces (OBJ uses 1-based indexing)
            for f in faces:
                obj_content += f"f {f[0]+1} {f[1]+1} {f[2]+1}\n"
            
            return obj_content, f"{mesh_name}.obj"
        
        elif format == 'stl':
            # STL ASCII format
            stl_content = f"solid {mesh_name}\n"
            
            for face in faces:
                v0, v1, v2 = [vertices[i] for i in face]
                
                # Calculate normal
                edge1 = np.array(v1) - np.array(v0)
                edge2 = np.array(v2) - np.array(v0)
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                stl_content += f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n"
                stl_content += f"    outer loop\n"
                stl_content += f"      vertex {v0[0]} {v0[1]} {v0[2]}\n"
                stl_content += f"      vertex {v1[0]} {v1[1]} {v1[2]}\n"
                stl_content += f"      vertex {v2[0]} {v2[1]} {v2[2]}\n"
                stl_content += f"    endloop\n"
                stl_content += f"  endfacet\n"
            
            stl_content += f"endsolid {mesh_name}\n"
            
            return stl_content, f"{mesh_name}.stl"
        
        elif format == 'json':
            # JSON format
            json_content = json.dumps(mesh_data, indent=2)
            return json_content, f"{mesh_name}.json"
        
    except Exception as e:
        st.error(f"Export error: {e}")
        return None, None

def process_batch_design(design_idea, index):
    """Process a single design idea in batch mode"""
    try:
        # Tier 1: Enhance prompt
        enhanced = enhance_prompt(design_idea)
        if not enhanced:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'Prompt enhancement failed'
            }
        
        # Tier 2: Generate mesh
        mesh_code = generate_mesh_code(enhanced)
        if not mesh_code:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'Mesh generation failed'
            }
        
        # Parse mesh
        mesh_data = parse_mesh_json(mesh_code)
        if not mesh_data:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'Mesh parsing failed',
                'raw_output': mesh_code
            }
        
        return {
            'index': index,
            'design_idea': design_idea,
            'status': 'success',
            'enhanced_prompt': enhanced,
            'mesh_data': mesh_data
        }
        
    except Exception as e:
        return {
            'index': index,
            'design_idea': design_idea,
            'status': 'failed',
            'error': str(e)
        }

# =================================================================================
# TAB 1: CREATE MESH CODES
# =================================================================================
def tab_create_mesh():
    st.header("🎨 Create Mesh Codes")
    st.markdown("*Generate production-ready 3D mesh codes using AI. Precision that rivals CAD designers.*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Design Input")
        
        # Add example prompts
        st.markdown("**💡 Quick Examples** (click to load):")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        examples = {
            "Water Bottle": "Design a professional 750ml sports water bottle with twist-off cap, ergonomic grip section with finger grooves in the middle, integrated carrying loop on cap, push-pull sport mouthpiece, volume measurement markers (250ml, 500ml, 750ml) embossed on the side, wide stable base (8cm diameter), and slight taper from base to neck for easy holding.",
            "Desk Lamp": "Create a modern minimalist desk lamp with adjustable arm, round weighted base (15cm diameter, 2cm thick), flexible gooseneck section (30cm long, 1.5cm diameter), conical fabric shade (12cm top diameter, 18cm bottom diameter, 15cm tall), touch-sensitive power button on base, cable management channel in base, and rubber feet.",
            "Phone Stand": "Design an adjustable phone stand with two-position angle support (45° and 60°), non-slip rubber contact points, cable management slot at the back (8mm wide), weighted base for stability (10cm x 7cm), foldable design with living hinge, fits phones 6-8cm wide, thickness accommodation up to 12mm with case.",
            "Coffee Mug": "Create a 400ml ceramic coffee mug with ergonomic D-shaped handle positioned at 45° angle, comfortable 2mm rounded rim, exterior with subtle geometric texture pattern for grip, stable circular base (7.5cm diameter) with slight recess, gentle inward taper (2°) from rim to base, handle dimensions 12mm x 15mm cross-section.",
            "Bookend": "Design a modern L-shaped bookend with 15cm tall vertical section, 12cm deep horizontal base, 10cm width, rounded corners (5mm radius), felt padding on bottom surface, decorative geometric cutout pattern on vertical face, 3mm thickness throughout, weighted base section (can be filled), minimal aesthetic.",
        }
        
        with example_col1:
            if st.button("💧 Water Bottle", use_container_width=True):
                st.session_state.example_prompt = examples["Water Bottle"]
        with example_col2:
            if st.button("💡 Desk Lamp", use_container_width=True):
                st.session_state.example_prompt = examples["Desk Lamp"]
        with example_col3:
            if st.button("📱 Phone Stand", use_container_width=True):
                st.session_state.example_prompt = examples["Phone Stand"]
        
        example_col4, example_col5 = st.columns(2)
        with example_col4:
            if st.button("☕ Coffee Mug", use_container_width=True):
                st.session_state.example_prompt = examples["Coffee Mug"]
        with example_col5:
            if st.button("📚 Bookend", use_container_width=True):
                st.session_state.example_prompt = examples["Bookend"]
        
        st.divider()
        
        # Get the prompt value
        prompt_value = st.session_state.get('example_prompt', '')
        
        user_prompt = st.text_area(
            "Describe your 3D design:",
            value=prompt_value,
            height=200,
            placeholder="Example: Design a professional water bottle with a twist cap, ergonomic grip section in the middle, carrying loop integrated into the cap, sport mouthpiece, volume markers on the side, and a wide stable base. Make it 750ml capacity.",
            help="Be specific about what you want. The AI will add technical details, but describe features, purpose, and rough dimensions.",
            key="user_prompt_input"
        )
        
        # Clear example after loading
        if 'example_prompt' in st.session_state and st.session_state.example_prompt == prompt_value:
            del st.session_state.example_prompt
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            generate_btn = st.button("🚀 Generate Mesh", type="primary", use_container_width=True)
        with col_btn2:
            if st.session_state.generated_meshes:
                regenerate_btn = st.button("🔄 More Detail", use_container_width=True, 
                                          help="Regenerate last design with more vertices and detail")
            else:
                regenerate_btn = False
        with col_btn3:
            clear_btn = st.button("🗑️ Clear All", use_container_width=True)
        
        if clear_btn:
            st.session_state.generated_meshes = []
            st.rerun()
        
        if regenerate_btn and st.session_state.generated_meshes:
            # Get the last generated mesh prompt
            last_prompt = st.session_state.generated_meshes[0]['prompt']
            enhanced_instruction = f"{last_prompt}\n\nIMPORTANT: Generate with MUCH MORE DETAIL - use at least 500-1000 vertices, add all curves, textures, and fine features described. Make it production-quality."
            
            with st.spinner("🔄 Regenerating with enhanced detail..."):
                enhanced_prompt = enhance_prompt(enhanced_instruction)
                
                if enhanced_prompt:
                    with st.spinner("🔄 Generating high-detail mesh..."):
                        mesh_code = generate_mesh_code(enhanced_prompt + "\n\nCRITICAL: Use at least 500 vertices minimum. Add fine geometric details.")
                        
                        if mesh_code:
                            mesh_data = parse_mesh_json(mesh_code)
                            
                            if mesh_data:
                                st.session_state.generated_meshes.insert(0, {
                                    'prompt': last_prompt + " [Enhanced Detail]",
                                    'enhanced': enhanced_prompt,
                                    'mesh_data': mesh_data,
                                    'raw_code': mesh_code,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                st.rerun()
        
        if generate_btn and user_prompt:
            with st.spinner("🔄 Tier 1: Enhancing your prompt..."):
                enhanced_prompt = enhance_prompt(user_prompt)
                
                if enhanced_prompt:
                    st.success("✅ Prompt enhanced successfully!")
                    
                    with st.expander("📝 View Enhanced Specifications"):
                        st.markdown(enhanced_prompt)
                    
                    with st.spinner("🔄 Tier 2: Generating production mesh code..."):
                        mesh_code = generate_mesh_code(enhanced_prompt)
                        
                        if mesh_code:
                            st.success("✅ Mesh code generated!")
                            
                            # Parse and validate
                            mesh_data = parse_mesh_json(mesh_code)
                            
                            if mesh_data:
                                st.session_state.generated_meshes.insert(0, {
                                    'prompt': user_prompt,
                                    'enhanced': enhanced_prompt,
                                    'mesh_data': mesh_data,
                                    'raw_code': mesh_code,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                st.success("✅ Mesh validated and ready!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to parse mesh code. Check raw output below.")
                                with st.expander("🔍 Raw Output"):
                                    st.code(mesh_code, language='json')
    
    with col2:
        st.subheader("Generated Meshes")
        
        if st.session_state.generated_meshes:
            for idx, mesh_item in enumerate(st.session_state.generated_meshes):
                with st.container():
                    st.markdown(f"**Design {idx + 1}** - {mesh_item['timestamp']}")
                    st.caption(f"*{mesh_item['prompt'][:100]}...*")
                    
                    # Render preview
                    fig = render_mesh_plotly(mesh_item['mesh_data'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Action buttons
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        if st.button(f"📄 Details", key=f"details_{idx}"):
                            st.session_state[f'show_details_{idx}'] = not st.session_state.get(f'show_details_{idx}', False)
                    
                    with col_b:
                        if st.button(f"💾 Export", key=f"export_{idx}"):
                            st.session_state[f'show_export_{idx}'] = not st.session_state.get(f'show_export_{idx}', False)
                    
                    with col_c:
                        if st.button(f"📋 Code", key=f"code_{idx}"):
                            st.session_state[f'show_code_{idx}'] = not st.session_state.get(f'show_code_{idx}', False)
                    
                    with col_d:
                        if st.button(f"🗑️ Delete", key=f"delete_{idx}"):
                            st.session_state.generated_meshes.pop(idx)
                            st.rerun()
                    
                    # Show details
                    if st.session_state.get(f'show_details_{idx}', False):
                        with st.expander("📊 Mesh Details", expanded=True):
                            mesh_data = mesh_item['mesh_data']
                            st.json({
                                'Mesh Name': mesh_data.get('mesh_name', 'N/A'),
                                'Vertices': len(mesh_data.get('vertices', [])),
                                'Faces': len(mesh_data.get('faces', [])),
                                'Metadata': mesh_data.get('metadata', {}),
                                'Materials': mesh_data.get('materials', {})
                            })
                    
                    # Show export options
                    if st.session_state.get(f'show_export_{idx}', False):
                        with st.expander("📥 Export Options", expanded=True):
                            export_format = st.selectbox(
                                "Format:",
                                ['obj', 'stl', 'json'],
                                key=f"export_format_{idx}"
                            )
                            
                            content, filename = export_mesh(mesh_item['mesh_data'], export_format)
                            
                            if content:
                                st.download_button(
                                    f"⬇️ Download {export_format.upper()}",
                                    content,
                                    filename,
                                    mime='text/plain',
                                    key=f"download_{idx}"
                                )
                    
                    # Show code
                    if st.session_state.get(f'show_code_{idx}', False):
                        with st.expander("💻 Mesh Code", expanded=True):
                            st.code(json.dumps(mesh_item['mesh_data'], indent=2), language='json')
                    
                    st.divider()
        else:
            st.info("💡 No meshes generated yet. Enter a design prompt and click Generate!")

# =================================================================================
# TAB 2: RENDER MODELS & DESIGN
# =================================================================================
def tab_render_models():
    st.header("🖼️ Render Models & Design")
    st.markdown("*Import, export, and render 3D models with multi-format support*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Import Model")
        
        upload_option = st.radio(
            "Import method:",
            ["Upload File", "Paste JSON Code", "Load from History"],
            horizontal=True
        )
        
        mesh_data_to_render = None
        
        if upload_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['json', 'obj', 'stl'],
                help="Currently supports JSON mesh format. OBJ/STL parsing coming soon."
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.json'):
                        content = uploaded_file.read().decode('utf-8')
                        mesh_data_to_render = json.loads(content)
                        st.success(f"✅ Loaded {uploaded_file.name}")
                    else:
                        st.warning("⚠️ OBJ/STL parsing not yet implemented. Use JSON format.")
                except Exception as e:
                    st.error(f"❌ Failed to load file: {e}")
        
        elif upload_option == "Paste JSON Code":
            json_input = st.text_area(
                "Paste mesh JSON code:",
                height=300,
                placeholder='{"mesh_name": "...", "vertices": [...], "faces": [...]}'
            )
            
            if st.button("📊 Parse & Render"):
                try:
                    mesh_data_to_render = json.loads(json_input)
                    st.success("✅ JSON parsed successfully!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {e}")
        
        elif upload_option == "Load from History":
            if st.session_state.generated_meshes:
                selected_idx = st.selectbox(
                    "Select a mesh:",
                    range(len(st.session_state.generated_meshes)),
                    format_func=lambda i: f"Design {i+1}: {st.session_state.generated_meshes[i]['prompt'][:50]}..."
                )
                
                if st.button("📥 Load Selected"):
                    mesh_data_to_render = st.session_state.generated_meshes[selected_idx]['mesh_data']
                    st.success("✅ Mesh loaded from history!")
            else:
                st.info("💡 No meshes in history. Generate some in the 'Create Mesh Codes' tab!")
    
    with col2:
        st.subheader("3D Viewer")
        
        if mesh_data_to_render:
            fig = render_mesh_plotly(mesh_data_to_render)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Mesh statistics
                st.markdown("### 📊 Mesh Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("Vertices", len(mesh_data_to_render.get('vertices', [])))
                
                with col_stat2:
                    st.metric("Faces", len(mesh_data_to_render.get('faces', [])))
                
                with col_stat3:
                    st.metric("Edges", len(mesh_data_to_render.get('vertices', [])) * 3 // 2)
                
                # Export section
                st.markdown("### 💾 Export Options")
                
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    content, filename = export_mesh(mesh_data_to_render, 'obj')
                    if content:
                        st.download_button(
                            "⬇️ OBJ",
                            content,
                            filename,
                            mime='text/plain',
                            use_container_width=True
                        )
                
                with col_exp2:
                    content, filename = export_mesh(mesh_data_to_render, 'stl')
                    if content:
                        st.download_button(
                            "⬇️ STL",
                            content,
                            filename,
                            mime='text/plain',
                            use_container_width=True
                        )
                
                with col_exp3:
                    content, filename = export_mesh(mesh_data_to_render, 'json')
                    if content:
                        st.download_button(
                            "⬇️ JSON",
                            content,
                            filename,
                            mime='application/json',
                            use_container_width=True
                        )
        else:
            st.info("👈 Import a model from the left panel to view it here")

# =================================================================================
# TAB 3: AI & AUTOMATIONS
# =================================================================================
def tab_ai_automations():
    st.header("🤖 AI & Automations")
    st.markdown("*Batch process multiple designs with parallel AI generation*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Design Ideas")
        
        # Sample CSV template
        st.markdown("**CSV Format Required:**")
        st.code("design_ideas\nCreate a modern chair with ergonomic backrest\nDesign a minimalist table lamp\nGenerate a decorative vase", language='csv')
        
        # Download template
        template_csv = "design_ideas\nCreate a modern chair with ergonomic backrest\nDesign a minimalist table lamp\nGenerate a decorative vase"
        st.download_button(
            "📥 Download Template",
            template_csv,
            "design_ideas_template.csv",
            mime='text/csv',
            use_container_width=True
        )
        
        st.divider()
        
        # File upload
        uploaded_csv = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV file must contain a 'design_ideas' column"
        )
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                
                if 'design_ideas' not in df.columns:
                    st.error("❌ CSV must contain a 'design_ideas' column!")
                else:
                    st.success(f"✅ Loaded {len(df)} design ideas")
                    
                    # Preview
                    st.markdown("**Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Processing options
                    st.divider()
                    st.subheader("⚙️ Processing Options")
                    
                    max_workers = st.slider(
                        "Parallel workers:",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Number of concurrent AI requests. Higher = faster but more resource intensive."
                    )
                    
                    process_btn = st.button(
                        f"🚀 Process {len(df)} Designs",
                        type="primary",
                        use_container_width=True
                    )
                    
                    if process_btn:
                        st.session_state.batch_results = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_container = st.empty()
                        
                        design_ideas = df['design_ideas'].tolist()
                        total = len(design_ideas)
                        completed = 0
                        
                        # Process in parallel
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_idx = {
                                executor.submit(process_batch_design, idea, idx): idx 
                                for idx, idea in enumerate(design_ideas)
                            }
                            
                            for future in as_completed(future_to_idx):
                                result = future.result()
                                st.session_state.batch_results.append(result)
                                
                                completed += 1
                                progress = completed / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {completed}/{total} designs completed")
                                
                                # Show intermediate results
                                success_count = sum(1 for r in st.session_state.batch_results if r['status'] == 'success')
                                failed_count = completed - success_count
                                
                                results_container.markdown(f"""
                                **Current Status:**
                                - ✅ Successful: {success_count}
                                - ❌ Failed: {failed_count}
                                - ⏳ Remaining: {total - completed}
                                """)
                        
                        status_text.text(f"✅ Completed! {success_count}/{total} designs generated successfully.")
                        st.balloons()
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
    
    with col2:
        st.subheader("📊 Batch Results")
        
        if st.session_state.batch_results:
            # Summary statistics
            total_results = len(st.session_state.batch_results)
            successful = sum(1 for r in st.session_state.batch_results if r['status'] == 'success')
            failed = total_results - successful
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total", total_results)
            with col_stat2:
                st.metric("✅ Success", successful)
            with col_stat3:
                st.metric("❌ Failed", failed)
            
            st.divider()
            
            # Export all results
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("💾 Export All Meshes (ZIP)", use_container_width=True):
                    st.info("ZIP export functionality coming soon!")
            
            with export_col2:
                # Create results CSV
                results_df = pd.DataFrame([
                    {
                        'index': r['index'],
                        'design_idea': r['design_idea'],
                        'status': r['status'],
                        'mesh_name': r.get('mesh_data', {}).get('mesh_name', 'N/A') if r['status'] == 'success' else 'N/A',
                        'vertices': len(r.get('mesh_data', {}).get('vertices', [])) if r['status'] == 'success' else 0,
                        'faces': len(r.get('mesh_data', {}).get('faces', [])) if r['status'] == 'success' else 0,
                        'error': r.get('error', '')
                    }
                    for r in st.session_state.batch_results
                ])
                
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Report (CSV)",
                    csv_data,
                    "batch_results.csv",
                    mime='text/csv',
                    use_container_width=True
                )
            
            st.divider()
            
            # Display individual results
            st.markdown("### Individual Results")
            
            filter_status = st.selectbox(
                "Filter by status:",
                ["All", "Success", "Failed"]
            )
            
            filtered_results = st.session_state.batch_results
            if filter_status == "Success":
                filtered_results = [r for r in filtered_results if r['status'] == 'success']
            elif filter_status == "Failed":
                filtered_results = [r for r in filtered_results if r['status'] == 'failed']
            
            for result in filtered_results:
                with st.expander(
                    f"{'✅' if result['status'] == 'success' else '❌'} Design {result['index'] + 1}: {result['design_idea'][:60]}...",
                    expanded=False
                ):
                    if result['status'] == 'success':
                        # Show mesh preview
                        fig = render_mesh_plotly(result['mesh_data'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Export options
                        col_e1, col_e2, col_e3 = st.columns(3)
                        
                        with col_e1:
                            content, filename = export_mesh(result['mesh_data'], 'obj')
                            if content:
                                st.download_button(
                                    "⬇️ OBJ",
                                    content,
                                    filename,
                                    key=f"batch_obj_{result['index']}"
                                )
                        
                        with col_e2:
                            content, filename = export_mesh(result['mesh_data'], 'stl')
                            if content:
                                st.download_button(
                                    "⬇️ STL",
                                    content,
                                    filename,
                                    key=f"batch_stl_{result['index']}"
                                )
                        
                        with col_e3:
                            content, filename = export_mesh(result['mesh_data'], 'json')
                            if content:
                                st.download_button(
                                    "⬇️ JSON",
                                    content,
                                    filename,
                                    key=f"batch_json_{result['index']}"
                                )
                    else:
                        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
                        
                        if 'raw_output' in result:
                            with st.expander("🔍 Raw AI Output"):
                                st.code(result['raw_output'])
        else:
            st.info("💡 Upload a CSV file and process designs to see results here")

# =================================================================================
# MAIN APP
# =================================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🎨 AI 3D Mesh Designer")
        st.markdown("---")
        
        st.markdown("### 💡 Pro Tips for Best Results")
        
        with st.expander("✅ What Makes a Good Prompt", expanded=False):
            st.markdown("""
            **DO Include:**
            - Specific features (handles, buttons, grips)
            - Functional elements (drainage, ventilation)
            - Rough dimensions or proportions
            - Material/finish suggestions
            - Purpose and use case
            
            **Example:**
            "Water bottle with sport cap, grip section, 750ml"
            
            **NOT:**
            "Make a bottle"
            """)
        
        with st.expander("🎯 Mesh Quality Guide", expanded=False):
            st.markdown("""
            **Quality Indicators:**
            - ✅ 200+ vertices: Professional
            - ⚠️ 50-200 vertices: Basic
            - ❌ <50 vertices: Too simple
            
            **If mesh is too simple:**
            Add to your prompt: "with detailed features, smooth curves, and textured surfaces"
            """)
        
        with st.expander("🚀 Speed Tips", expanded=False):
            st.markdown("""
            **Batch Processing:**
            - Use 3-5 parallel workers
            - Start with 10 designs to test
            - Detailed prompts = better results
            
            **Single Designs:**
            - More detail in prompt = better mesh
            - Use example buttons to start
            - Review enhanced specs before generation
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Features")
        st.markdown("""
        - ✨ AI-powered mesh generation
        - 🎨 Production-ready designs
        - 🔄 2-tier enhancement system
        - 📊 Multi-format export
        - 🚀 Batch processing
        - 🖼️ Interactive 3D rendering
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ API Status")
        
        account_id, auth_token = get_cloudflare_api_config()
        
        if account_id and auth_token:
            st.success("✅ API Configured")
        else:
            st.error("❌ API Not Configured")
            st.markdown("""
            **Setup Required:**
            1. Add to `.streamlit/secrets.toml`:
            ```toml
            CLOUDFLARE_ACCOUNT_ID = "your-id"
            CLOUDFLARE_AUTH_TOKEN = "your-token"
            ```
            """)
        
        st.markdown("---")
        st.caption("Powered by Cloudflare AI")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🎨 Create Mesh Codes",
        "🖼️ Render Models & Design",
        "🤖 AI & Automations"
    ])
    
    with tab1:
        tab_create_mesh()
    
    with tab2:
        tab_render_models()
    
    with tab3:
        tab_ai_automations()

if __name__ == "__main__":
    main()
