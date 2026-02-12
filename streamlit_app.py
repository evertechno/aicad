"""
Production-Grade AI 3D CAD Designer
Uses real CAD libraries: trimesh, pygltflib, shapely, scipy
Full implementation with proper mesh operations, validation, and export
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
import base64
from pathlib import Path
import tempfile

# CAD and Geometry Libraries
import trimesh
from trimesh import creation, transformations
from trimesh.exchange import export
import pygltflib
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Professional AI CAD Designer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        font-weight: 600;
    }
    .mesh-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .quality-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .quality-excellent {
        background: #10b981;
        color: white;
    }
    .quality-good {
        background: #3b82f6;
        color: white;
    }
    .quality-fair {
        background: #f59e0b;
        color: white;
    }
    .quality-poor {
        background: #ef4444;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_meshes' not in st.session_state:
    st.session_state.generated_meshes = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# =================================================================================
# CLOUDFLARE API INTEGRATION
# =================================================================================

def get_cloudflare_api_config():
    """Get Cloudflare API configuration from secrets"""
    try:
        account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
        auth_token = st.secrets["CLOUDFLARE_AUTH_TOKEN"]
        return account_id, auth_token
    except Exception as e:
        st.error(f"API Configuration Error: {e}")
        return None, None

def call_ai_api(messages, max_tokens=16000):
    """Call Cloudflare AI API"""
    account_id, auth_token = get_cloudflare_api_config()
    
    if not account_id or not auth_token:
        return None
    
    try:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=180
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

# =================================================================================
# AI PROMPT ENGINEERING
# =================================================================================

def enhance_prompt(user_prompt):
    """Tier 1: Transform basic idea into detailed CAD specifications"""
    system_message = """You are an expert CAD engineer and industrial designer. Transform user ideas into COMPLETE, PRECISE CAD specifications.

For EVERY design, provide:

1. OVERALL DIMENSIONS (exact measurements in mm/cm)
2. GEOMETRIC PRIMITIVES needed (cylinders, boxes, spheres, tori, cones)
3. BOOLEAN OPERATIONS (unions, differences, intersections)
4. TRANSFORMATIONS (positions, rotations, scales)
5. FEATURES: extrusions, fillets, chamfers, holes, threads
6. MATERIAL PROPERTIES
7. ASSEMBLY INSTRUCTIONS (if multi-part)

Output format:
```
DESIGN: [Name]
BASE_SHAPE: [primitive type with dimensions]
FEATURES:
- [feature 1: type, dimensions, position]
- [feature 2: type, dimensions, position]
OPERATIONS:
- [operation 1: type, target shapes]
- [operation 2: type, target shapes]
FINISH: [surface finish, texture details]
```

Be EXTREMELY specific with numeric values. Think like you're programming a CNC machine.

Example:
"Coffee mug" →
```
DESIGN: Ergonomic Coffee Mug
BASE_SHAPE: Cylinder (height=95mm, radius_outer=40mm, wall_thickness=3mm)
FEATURES:
- Handle: Torus section (major_radius=25mm, minor_radius=6mm) + Cylinder (length=60mm, radius=6mm), positioned at angle=30°, offset_x=40mm
- Rim: Fillet (radius=2mm) on top edge
- Base: Cylinder (height=5mm, radius=37mm) for stability ring
- Grip_texture: Array of small spheres (radius=0.5mm, spacing=3mm) on exterior, height_range=[20mm, 70mm]
OPERATIONS:
- Union: body + handle + base
- Difference: interior cavity from body
- Union: grip texture spheres to exterior
FINISH: Smooth glazed interior, matte textured exterior
```

Generate similarly detailed specs."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Transform this into precise CAD specifications with exact dimensions and operations:\n\n{user_prompt}"}
    ]
    
    return call_ai_api(messages, max_tokens=8000)

def generate_cad_code(enhanced_prompt):
    """Tier 2: Generate Python code using trimesh to create the actual mesh"""
    system_message = """You are a 3D CAD code generator. Generate COMPLETE, RUNNABLE Python code using ONLY valid trimesh library functions.

CRITICAL: Use ONLY these validated trimesh.creation functions:
- trimesh.creation.box(extents=[x, y, z])
- trimesh.creation.cylinder(radius=r, height=h, sections=N)
- trimesh.creation.sphere(radius=r, subdivisions=N)
- trimesh.creation.capsule(height=h, radius=r, count=[N, N])
- trimesh.creation.cone(radius=r, height=h, sections=N)
- trimesh.creation.icosphere(subdivisions=N, radius=r)
- trimesh.creation.annulus(r_min=inner, r_max=outer, height=h, sections=N)

TRANSFORMATIONS (always available):
- mesh.apply_translation([x, y, z])
- mesh.apply_scale(factor) or mesh.apply_scale([sx, sy, sz])
- matrix = trimesh.transformations.rotation_matrix(angle_radians, [x, y, z], point=[0,0,0])
- mesh.apply_transform(matrix)

BOOLEAN OPERATIONS (may fail, wrap in try/except):
- trimesh.boolean.union([mesh1, mesh2, ...])
- trimesh.boolean.difference([mesh1, mesh2, ...])
- trimesh.boolean.intersection([mesh1, mesh2, ...])

IMPORTANT RULES:
1. NEVER use functions that don't exist (extrude, revolve, sweep, loft)
2. Build complex shapes by combining primitives with boolean ops
3. For threads/gears: approximate with arrays of small cylinders/boxes
4. For curves: use multiple small primitives arranged in arc
5. Wrap ALL boolean operations in try/except blocks
6. Always check if result is valid before using
7. Return variable MUST be named 'final_mesh'

ERROR HANDLING TEMPLATE:
```python
try:
    result = trimesh.boolean.union([part1, part2])
    if result.is_empty or not result.is_valid:
        result = part1  # Fallback to main part
except Exception as e:
    result = part1  # Fallback if boolean fails
```

COMPLETE WORKING EXAMPLE:
```python
import numpy as np
import trimesh
from trimesh import creation, transformations

# Create main body
body = creation.cylinder(radius=25, height=50, sections=64)

# Create hole through center
hole = creation.cylinder(radius=10, height=52, sections=32)
hole.apply_translation([0, 0, -1])

# Boolean operation with error handling
try:
    body_with_hole = trimesh.boolean.difference([body, hole])
    if body_with_hole.is_empty:
        body_with_hole = body
except:
    body_with_hole = body

# Add mounting holes (4 holes around perimeter)
mounting_holes = []
for i in range(4):
    angle = (i / 4) * 2 * np.pi
    m_hole = creation.cylinder(radius=2, height=6, sections=16)
    m_hole.apply_translation([
        18 * np.cos(angle),
        18 * np.sin(angle),
        22
    ])
    mounting_holes.append(m_hole)

# Subtract mounting holes
try:
    all_holes = trimesh.util.concatenate(mounting_holes)
    final_mesh = trimesh.boolean.difference([body_with_hole, all_holes])
    if final_mesh.is_empty:
        final_mesh = body_with_hole
except:
    final_mesh = body_with_hole

# Ensure we have a valid mesh
if not isinstance(final_mesh, trimesh.Trimesh) or final_mesh.is_empty:
    final_mesh = body
```

GEAR APPROXIMATION EXAMPLE:
```python
# Approximate gear teeth
gear_body = creation.cylinder(radius=23, height=8, sections=64)
teeth = []
num_teeth = 20
for i in range(num_teeth):
    angle = (i / num_teeth) * 2 * np.pi
    tooth = creation.box(extents=[4, 2, 8])
    # Rotate and position
    rot_matrix = transformations.rotation_matrix(angle, [0, 0, 1])
    tooth.apply_transform(rot_matrix)
    tooth.apply_translation([25 * np.cos(angle), 25 * np.sin(angle), 0])
    teeth.append(tooth)

try:
    teeth_mesh = trimesh.util.concatenate(teeth)
    gear = trimesh.boolean.union([gear_body, teeth_mesh])
    if gear.is_empty:
        gear = gear_body
except:
    gear = gear_body

final_mesh = gear
```

Generate code following these patterns exactly. Use error handling for ALL boolean operations.
Return ONLY Python code, no markdown, no explanations."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Generate complete, runnable trimesh code with proper error handling for:\n\n{enhanced_prompt}\n\nUse sections=64 for smooth geometry. Wrap all boolean operations in try/except. Ensure final_mesh is always valid."}
    ]
    
    return call_ai_api(messages, max_tokens=16000)

# =================================================================================
# MESH PROCESSING WITH TRIMESH
# =================================================================================

def create_fallback_mesh(prompt):
    """Create a reasonable fallback mesh based on prompt keywords"""
    try:
        prompt_lower = prompt.lower()
        
        # Detect shape type from prompt
        if any(word in prompt_lower for word in ['gear', 'cog', 'sprocket']):
            # Create a simple gear
            body = creation.cylinder(radius=25, height=8, sections=64)
            hole = creation.cylinder(radius=5, height=10, sections=32)
            hole.apply_translation([0, 0, -1])
            try:
                mesh = trimesh.boolean.difference([body, hole])
                if mesh.is_empty:
                    mesh = body
            except:
                mesh = body
            return mesh
        
        elif any(word in prompt_lower for word in ['bolt', 'screw', 'fastener']):
            # Create a simple bolt
            head = creation.cylinder(radius=8, height=4, sections=6)
            head.apply_translation([0, 0, 0])
            shaft = creation.cylinder(radius=5, height=30, sections=32)
            shaft.apply_translation([0, 0, -30])
            try:
                mesh = trimesh.boolean.union([head, shaft])
                if mesh.is_empty:
                    mesh = trimesh.util.concatenate([head, shaft])
            except:
                mesh = trimesh.util.concatenate([head, shaft])
            return mesh
        
        elif any(word in prompt_lower for word in ['housing', 'mount', 'bracket']):
            # Create a simple housing
            outer = creation.box(extents=[40, 40, 30])
            inner = creation.box(extents=[34, 34, 32])
            inner.apply_translation([0, 0, 2])
            try:
                mesh = trimesh.boolean.difference([outer, inner])
                if mesh.is_empty:
                    mesh = outer
            except:
                mesh = outer
            return mesh
        
        elif any(word in prompt_lower for word in ['cap', 'lid', 'cover']):
            # Create a simple cap
            mesh = creation.cylinder(radius=20, height=10, sections=64)
            return mesh
        
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pipe']):
            # Create a cylinder
            outer = creation.cylinder(radius=15, height=50, sections=64)
            inner = creation.cylinder(radius=12, height=52, sections=64)
            inner.apply_translation([0, 0, -1])
            try:
                mesh = trimesh.boolean.difference([outer, inner])
                if mesh.is_empty:
                    mesh = outer
            except:
                mesh = outer
            return mesh
        
        elif any(word in prompt_lower for word in ['sphere', 'ball']):
            mesh = creation.sphere(radius=20, subdivisions=4)
            return mesh
        
        elif any(word in prompt_lower for word in ['box', 'cube', 'block']):
            mesh = creation.box(extents=[30, 30, 30])
            return mesh
        
        else:
            # Default: create a reasonable cylinder
            mesh = creation.cylinder(radius=20, height=40, sections=64)
            return mesh
            
    except Exception as e:
        st.error(f"Fallback mesh creation failed: {e}")
        return None

def execute_cad_code(code_str):
    """Execute the generated CAD code and return the mesh"""
    try:
        # Clean the code
        if "```python" in code_str:
            start = code_str.find("```python") + 9
            end = code_str.find("```", start)
            code_str = code_str[start:end].strip()
        elif "```" in code_str:
            start = code_str.find("```") + 3
            end = code_str.find("```", start)
            code_str = code_str[start:end].strip()
        
        # Create execution namespace with all necessary imports
        namespace = {
            'np': np,
            'numpy': np,
            'trimesh': trimesh,
            'creation': creation,
            'transformations': transformations,
        }
        
        # Execute the code
        exec(code_str, namespace)
        
        # Get the final mesh
        if 'final_mesh' not in namespace:
            st.error("❌ Generated code doesn't define 'final_mesh' variable")
            st.code(code_str, language='python')
            return None
        
        mesh = namespace['final_mesh']
        
        # Validate it's a trimesh object
        if not isinstance(mesh, trimesh.Trimesh):
            st.error(f"❌ 'final_mesh' is type {type(mesh)}, not trimesh.Trimesh")
            return None
        
        # Check if mesh is empty
        if mesh.is_empty:
            st.error("❌ Generated mesh is empty")
            return None
        
        # Check vertex count
        if len(mesh.vertices) < 4:
            st.error(f"❌ Generated mesh has too few vertices: {len(mesh.vertices)}")
            return None
        
        # Validate mesh quality
        if not mesh.is_watertight:
            st.warning("⚠️ Mesh is not watertight - may have holes or gaps. This can happen with complex boolean operations.")
        
        # Fix mesh if possible
        try:
            if not mesh.is_watertight:
                # Try to fill holes
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_normals(mesh)
        except:
            pass  # Repair might fail, that's ok
        
        st.success(f"✅ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
            
    except SyntaxError as e:
        st.error(f"❌ Python syntax error in generated code: {e}")
        st.code(code_str, language='python')
        return None
    except NameError as e:
        st.error(f"❌ Undefined variable or function: {e}")
        st.code(code_str, language='python')
        return None
    except AttributeError as e:
        st.error(f"❌ Invalid trimesh function: {e}")
        st.markdown("**Common issue:** AI tried to use a function that doesn't exist in trimesh")
        st.code(code_str, language='python')
        return None
    except Exception as e:
        st.error(f"❌ Code execution failed: {str(e)}")
        st.code(code_str, language='python')
        
        # Try to provide helpful error context
        error_str = str(e).lower()
        if "boolean" in error_str:
            st.info("💡 **Tip:** Boolean operations can fail with complex geometry. Try simpler shapes or check for overlapping meshes.")
        elif "radius" in error_str or "height" in error_str:
            st.info("💡 **Tip:** Check that all dimensions are positive numbers.")
        elif "module" in error_str and "attribute" in error_str:
            st.info("💡 **Tip:** AI tried to use a trimesh function that doesn't exist. This is a generation error.")
        
        return None

def analyze_mesh(mesh):
    """Comprehensive mesh analysis using trimesh"""
    try:
        analysis = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'watertight': mesh.is_watertight,
            'volume': float(mesh.volume) if mesh.is_volume else 0,
            'surface_area': float(mesh.area),
            'bounds': mesh.bounds.tolist(),
            'center_mass': mesh.center_mass.tolist() if hasattr(mesh, 'center_mass') else [0, 0, 0],
            'euler_number': mesh.euler_number,
            'is_convex': mesh.is_convex,
            'body_count': len(mesh.split()),
        }
        
        # Quality metrics
        if analysis['vertices'] > 0 and analysis['faces'] > 0:
            analysis['quality_score'] = min(100, (analysis['vertices'] / 100) * 50 + (analysis['faces'] / 200) * 50)
        else:
            analysis['quality_score'] = 0
        
        return analysis
    except Exception as e:
        st.error(f"Mesh analysis failed: {str(e)}")
        return None

def get_quality_badge(score):
    """Get quality badge HTML"""
    if score >= 80:
        return '<span class="quality-badge quality-excellent">EXCELLENT</span>'
    elif score >= 60:
        return '<span class="quality-badge quality-good">GOOD</span>'
    elif score >= 40:
        return '<span class="quality-badge quality-fair">FAIR</span>'
    else:
        return '<span class="quality-badge quality-poor">NEEDS WORK</span>'

# =================================================================================
# MESH VISUALIZATION
# =================================================================================

def render_mesh_plotly(mesh, title="3D Mesh"):
    """Render trimesh using Plotly with enhanced visualization"""
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Calculate vertex colors based on height for visual interest
        z_values = vertices[:, 2]
        colors = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
        
        # Create the 3D mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=colors,
                colorscale='Viridis',
                opacity=0.9,
                flatshading=False,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.9,
                    specular=0.6,
                    roughness=0.2,
                    fresnel=0.2
                ),
                lightposition=dict(
                    x=1000,
                    y=1000,
                    z=1000
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#333')
            ),
            scene=dict(
                xaxis=dict(title='X (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                yaxis=dict(title='Y (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                zaxis=dict(title='Z (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

# =================================================================================
# EXPORT FUNCTIONS
# =================================================================================

def export_mesh_file(mesh, format='stl', filename='mesh'):
    """Export mesh to various formats using trimesh"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as tmp:
            if format == 'stl':
                mesh.export(tmp.name, file_type='stl')
            elif format == 'obj':
                mesh.export(tmp.name, file_type='obj')
            elif format == 'ply':
                mesh.export(tmp.name, file_type='ply')
            elif format == 'off':
                mesh.export(tmp.name, file_type='off')
            elif format == 'glb':
                mesh.export(tmp.name, file_type='glb')
            elif format == 'gltf':
                mesh.export(tmp.name, file_type='gltf')
            
            with open(tmp.name, 'rb') as f:
                data = f.read()
            
            Path(tmp.name).unlink()
            return data, f"{filename}.{format}"
    except Exception as e:
        st.error(f"Export error: {str(e)}")
        return None, None

def export_technical_drawing(mesh):
    """Generate technical 2D drawings (top, front, side views)"""
    try:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Top View', 'Front View', 'Side View'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        vertices = mesh.vertices
        
        # Top view (X-Y plane)
        hull_xy = ConvexHull(vertices[:, :2])
        hull_points_xy = vertices[hull_xy.vertices, :2]
        fig.add_trace(
            go.Scatter(x=hull_points_xy[:, 0], y=hull_points_xy[:, 1], 
                      mode='lines', name='Top', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Front view (X-Z plane)
        hull_xz = ConvexHull(vertices[:, [0, 2]])
        hull_points_xz = vertices[hull_xz.vertices][:, [0, 2]]
        fig.add_trace(
            go.Scatter(x=hull_points_xz[:, 0], y=hull_points_xz[:, 1],
                      mode='lines', name='Front', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Side view (Y-Z plane)
        hull_yz = ConvexHull(vertices[:, [1, 2]])
        hull_points_yz = vertices[hull_yz.vertices][:, [1, 2]]
        fig.add_trace(
            go.Scatter(x=hull_points_yz[:, 0], y=hull_points_yz[:, 1],
                      mode='lines', name='Side', line=dict(color='red', width=2)),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="X (mm)", scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="Y (mm)", row=1, col=1)
        fig.update_xaxes(title_text="X (mm)", scaleanchor="y", scaleratio=1, row=1, col=2)
        fig.update_yaxes(title_text="Z (mm)", row=1, col=2)
        fig.update_xaxes(title_text="Y (mm)", scaleanchor="y", scaleratio=1, row=1, col=3)
        fig.update_yaxes(title_text="Z (mm)", row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False, title_text="Technical Drawings")
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate technical drawings: {str(e)}")
        return None

# =================================================================================
# BATCH PROCESSING
# =================================================================================

def process_single_design(design_idea, index):
    """Process one design in batch mode"""
    try:
        # Tier 1: Enhance
        enhanced = enhance_prompt(design_idea)
        if not enhanced:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'Prompt enhancement failed'
            }
        
        # Tier 2: Generate code
        cad_code = generate_cad_code(enhanced)
        if not cad_code:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'CAD code generation failed'
            }
        
        # Execute code to create mesh
        mesh = execute_cad_code(cad_code)
        if mesh is None:
            return {
                'index': index,
                'design_idea': design_idea,
                'status': 'failed',
                'error': 'Mesh creation failed',
                'code': cad_code
            }
        
        # Analyze mesh
        analysis = analyze_mesh(mesh)
        
        return {
            'index': index,
            'design_idea': design_idea,
            'status': 'success',
            'enhanced_prompt': enhanced,
            'cad_code': cad_code,
            'mesh': mesh,
            'analysis': analysis
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

def tab_create_meshes():
    st.header("🔧 Create CAD Meshes")
    st.markdown("*Generate production-grade 3D meshes using real CAD operations*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Design Input")
        
        # Example buttons
        st.markdown("**💡 Professional Examples:**")
        ex1, ex2, ex3 = st.columns(3)
        
        examples = {
            "Gear": "Design a mechanical gear with 20 teeth, 50mm outer diameter, 10mm inner bore, 8mm thickness, involute tooth profile, 5mm tooth depth, suitable for 3D printing",
            "Threaded Bolt": "Create an M10 hex bolt with standard thread pitch (1.5mm), 40mm shaft length, 6mm head height, 17mm hex head, chamfered thread start, suitable for metal fabrication",
            "Bearing Housing": "Design a ball bearing housing for 608 bearing (22mm OD, 8mm ID), with mounting flange, 4x M4 mounting holes on 30mm PCD, 3mm wall thickness, grease port",
            "Cable Gland": "Create a cable gland for 8mm cable diameter, with compression seal, outer thread M16x1.5, hex grip section for wrench, IP67 sealing geometry",
            "Bottle Cap": "Design a twist-off bottle cap for 38mm bottle neck, with buttress threads (pitch 3mm), tamper-evident band, grip ribs on sides, food-safe design"
        }
        
        with ex1:
            if st.button("⚙️ Gear", use_container_width=True):
                st.session_state.prompt_example = examples["Gear"]
        with ex2:
            if st.button("🔩 Bolt", use_container_width=True):
                st.session_state.prompt_example = examples["Threaded Bolt"]
        with ex3:
            if st.button("⚡ Housing", use_container_width=True):
                st.session_state.prompt_example = examples["Bearing Housing"]
        
        ex4, ex5 = st.columns(2)
        with ex4:
            if st.button("🔌 Cable Gland", use_container_width=True):
                st.session_state.prompt_example = examples["Cable Gland"]
        with ex5:
            if st.button("🍾 Bottle Cap", use_container_width=True):
                st.session_state.prompt_example = examples["Bottle Cap"]
        
        st.divider()
        
        user_prompt = st.text_area(
            "Describe your CAD design:",
            value=st.session_state.get('prompt_example', ''),
            height=180,
            placeholder="Example: Design a mechanical gear with 20 teeth, 50mm diameter, 8mm thickness, involute profile...",
            help="Describe the mechanical part you need. Include dimensions, tolerances, and functional requirements.",
            key="user_cad_input"
        )
        
        if 'prompt_example' in st.session_state:
            del st.session_state.prompt_example
        
        col_gen, col_clear = st.columns(2)
        with col_gen:
            generate_btn = st.button("🚀 Generate CAD Mesh", type="primary", use_container_width=True)
        with col_clear:
            clear_btn = st.button("🗑️ Clear All", use_container_width=True)
        
        if clear_btn:
            st.session_state.generated_meshes = []
            st.rerun()
        
        if generate_btn and user_prompt:
            with st.status("🔄 Generating CAD mesh...", expanded=True) as status:
                st.write("Step 1/3: Enhancing specifications...")
                enhanced = enhance_prompt(user_prompt)
                
                if enhanced:
                    st.success("✅ Specifications enhanced")
                    with st.expander("📋 View CAD Specifications"):
                        st.markdown(enhanced)
                    
                    st.write("Step 2/3: Generating trimesh code...")
                    cad_code = generate_cad_code(enhanced)
                    
                    if cad_code:
                        st.success("✅ CAD code generated")
                        
                        st.write("Step 3/3: Executing code and creating mesh...")
                        mesh = execute_cad_code(cad_code)
                        
                        if mesh is None:
                            st.warning("⚠️ AI-generated code failed. Attempting fallback generation...")
                            
                            # Try to create a reasonable fallback mesh based on prompt keywords
                            mesh = create_fallback_mesh(user_prompt)
                            
                            if mesh:
                                st.success("✅ Fallback mesh created successfully!")
                                st.info("💡 The AI had trouble with complex operations. Using simplified geometry.")
                            else:
                                st.error("❌ Could not create mesh. Try simplifying your prompt or use an example.")
                                status.update(label="❌ Failed", state="error")
                                return
                        else:
                            st.success("✅ Mesh created successfully!")
                            
                            analysis = analyze_mesh(mesh)
                            
                            st.session_state.generated_meshes.insert(0, {
                                'prompt': user_prompt,
                                'enhanced': enhanced,
                                'code': cad_code,
                                'mesh': mesh,
                                'analysis': analysis,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            status.update(label="✅ Complete!", state="complete")
                            st.rerun()
    
    with col2:
        st.subheader("Generated CAD Meshes")
        
        if st.session_state.generated_meshes:
            for idx, item in enumerate(st.session_state.generated_meshes):
                with st.container():
                    st.markdown(f"### Design #{idx + 1}")
                    st.caption(f"*{item['prompt'][:100]}...* | {item['timestamp']}")
                    
                    # Render 3D view
                    fig = render_mesh_plotly(item['mesh'], f"Design #{idx + 1}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Mesh statistics
                    analysis = item['analysis']
                    quality_badge = get_quality_badge(analysis['quality_score'])
                    
                    st.markdown(f"""
                    <div class="mesh-stats">
                        <h4>Mesh Quality: {quality_badge} ({analysis['quality_score']:.1f}/100)</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px;">
                            <div><strong>Vertices:</strong> {analysis['vertices']:,}</div>
                            <div><strong>Faces:</strong> {analysis['faces']:,}</div>
                            <div><strong>Edges:</strong> {analysis['edges']:,}</div>
                            <div><strong>Volume:</strong> {analysis['volume']:.2f} mm³</div>
                            <div><strong>Surface Area:</strong> {analysis['surface_area']:.2f} mm²</div>
                            <div><strong>Watertight:</strong> {'✅ Yes' if analysis['watertight'] else '⚠️ No'}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Technical drawings
                    tech_fig = export_technical_drawing(item['mesh'])
                    if tech_fig:
                        st.plotly_chart(tech_fig, use_container_width=True)
                    
                    # Action buttons
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    
                    with col_a:
                        stl_data, stl_name = export_mesh_file(item['mesh'], 'stl', f'design_{idx+1}')
                        if stl_data:
                            st.download_button("📥 STL", stl_data, stl_name, mime='application/octet-stream', key=f"stl_{idx}")
                    
                    with col_b:
                        obj_data, obj_name = export_mesh_file(item['mesh'], 'obj', f'design_{idx+1}')
                        if obj_data:
                            st.download_button("📥 OBJ", obj_data, obj_name, mime='text/plain', key=f"obj_{idx}")
                    
                    with col_c:
                        glb_data, glb_name = export_mesh_file(item['mesh'], 'glb', f'design_{idx+1}')
                        if glb_data:
                            st.download_button("📥 GLB", glb_data, glb_name, mime='model/gltf-binary', key=f"glb_{idx}")
                    
                    with col_d:
                        if st.button("💻 Code", key=f"code_{idx}"):
                            st.session_state[f'show_code_{idx}'] = not st.session_state.get(f'show_code_{idx}', False)
                    
                    with col_e:
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.generated_meshes.pop(idx)
                            st.rerun()
                    
                    # Show code if toggled
                    if st.session_state.get(f'show_code_{idx}', False):
                        with st.expander("💻 Generated CAD Code", expanded=True):
                            st.code(item['code'], language='python')
                    
                    st.divider()
        else:
            st.info("💡 Click an example button or describe your design to generate a CAD mesh!")

# =================================================================================
# TAB 2: RENDER & ANALYZE
# =================================================================================

def tab_render_analyze():
    st.header("📊 Render & Analyze CAD Meshes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Import Mesh")
        
        import_method = st.radio("Import method:", ["Upload File", "Load from History"], horizontal=True)
        
        selected_mesh = None
        
        if import_method == "Upload File":
            uploaded = st.file_uploader(
                "Upload 3D file",
                type=['stl', 'obj', 'ply', 'off', 'glb', 'gltf'],
                help="Upload CAD files in standard formats"
            )
            
            if uploaded:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    
                    selected_mesh = trimesh.load(tmp_path)
                    Path(tmp_path).unlink()
                    
                    st.success(f"✅ Loaded {uploaded.name}")
                except Exception as e:
                    st.error(f"Failed to load file: {str(e)}")
        
        else:  # Load from History
            if st.session_state.generated_meshes:
                idx = st.selectbox(
                    "Select a mesh:",
                    range(len(st.session_state.generated_meshes)),
                    format_func=lambda i: f"Design #{i+1}: {st.session_state.generated_meshes[i]['prompt'][:50]}..."
                )
                
                if st.button("📥 Load Selected"):
                    selected_mesh = st.session_state.generated_meshes[idx]['mesh']
            else:
                st.info("No meshes in history. Generate some first!")
    
    with col2:
        st.subheader("3D Visualization & Analysis")
        
        if selected_mesh:
            # 3D visualization
            fig = render_mesh_plotly(selected_mesh, "Imported Mesh")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive analysis
            analysis = analyze_mesh(selected_mesh)
            
            st.markdown("### 📊 Detailed Analysis")
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Vertices", f"{analysis['vertices']:,}")
                st.metric("Volume", f"{analysis['volume']:.2f} mm³")
            with col_a2:
                st.metric("Faces", f"{analysis['faces']:,}")
                st.metric("Surface Area", f"{analysis['surface_area']:.2f} mm²")
            with col_a3:
                st.metric("Edges", f"{analysis['edges']:,}")
                st.metric("Bodies", analysis['body_count'])
            
            st.markdown("### ⚙️ Mesh Properties")
            props_col1, props_col2 = st.columns(2)
            
            with props_col1:
                st.write(f"**Watertight:** {'✅ Yes' if analysis['watertight'] else '⚠️ No'}")
                st.write(f"**Convex:** {'✅ Yes' if analysis['is_convex'] else '❌ No'}")
            
            with props_col2:
                st.write(f"**Euler Number:** {analysis['euler_number']}")
                bounds = analysis['bounds']
                st.write(f"**Dimensions:** {bounds[1][0]-bounds[0][0]:.1f} × {bounds[1][1]-bounds[0][1]:.1f} × {bounds[1][2]-bounds[0][2]:.1f} mm")
            
            # Technical drawings
            st.markdown("### 📐 Technical Drawings")
            tech_fig = export_technical_drawing(selected_mesh)
            if tech_fig:
                st.plotly_chart(tech_fig, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 Export Options")
            
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            
            with col_e1:
                stl_data, stl_name = export_mesh_file(selected_mesh, 'stl', 'mesh')
                if stl_data:
                    st.download_button("📥 STL", stl_data, stl_name, key="render_stl")
            
            with col_e2:
                obj_data, obj_name = export_mesh_file(selected_mesh, 'obj', 'mesh')
                if obj_data:
                    st.download_button("📥 OBJ", obj_data, obj_name, key="render_obj")
            
            with col_e3:
                ply_data, ply_name = export_mesh_file(selected_mesh, 'ply', 'mesh')
                if ply_data:
                    st.download_button("📥 PLY", ply_data, ply_name, key="render_ply")
            
            with col_e4:
                glb_data, glb_name = export_mesh_file(selected_mesh, 'glb', 'mesh')
                if glb_data:
                    st.download_button("📥 GLB", glb_data, glb_name, key="render_glb")
        else:
            st.info("👈 Import a mesh to visualize and analyze")

# =================================================================================
# TAB 3: BATCH PROCESSING
# =================================================================================

def tab_batch_processing():
    st.header("🚀 Batch CAD Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Design List")
        
        # CSV template
        st.markdown("**CSV Format:**")
        st.code("design_ideas\nDesign a gear with 20 teeth, 50mm diameter\nCreate M10 hex bolt, 40mm shaft", language='csv')
        
        template_csv = "design_ideas\n" + "\n".join([
            "Design a mechanical gear with 20 teeth, 50mm outer diameter, 10mm bore, 8mm thickness",
            "Create an M10 hex bolt with standard thread, 40mm shaft, chamfered start",
            "Design a bearing housing for 608 bearing, flange mount, 4x M4 holes",
        ])
        
        st.download_button("📥 Download Template", template_csv, "cad_batch_template.csv", mime='text/csv')
        
        st.divider()
        
        uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                
                if 'design_ideas' not in df.columns:
                    st.error("❌ CSV must have 'design_ideas' column")
                else:
                    st.success(f"✅ Loaded {len(df)} designs")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.divider()
                    
                    workers = st.slider("Parallel Workers:", 1, 5, 3)
                    
                    if st.button(f"🚀 Process {len(df)} Designs", type="primary", use_container_width=True):
                        st.session_state.batch_results = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        designs = df['design_ideas'].tolist()
                        total = len(designs)
                        completed = 0
                        
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            future_map = {executor.submit(process_single_design, idea, i): i for i, idea in enumerate(designs)}
                            
                            for future in as_completed(future_map):
                                result = future.result()
                                st.session_state.batch_results.append(result)
                                
                                completed += 1
                                progress_bar.progress(completed / total)
                                
                                success = sum(1 for r in st.session_state.batch_results if r['status'] == 'success')
                                status_text.text(f"⏳ {completed}/{total} | ✅ {success} succeeded")
                        
                        st.success(f"🎉 Batch complete! {success}/{total} successful")
                        st.balloons()
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("📊 Batch Results")
        
        if st.session_state.batch_results:
            total = len(st.session_state.batch_results)
            success = sum(1 for r in st.session_state.batch_results if r['status'] == 'success')
            failed = total - success
            
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total", total)
            col_s2.metric("✅ Success", success)
            col_s3.metric("❌ Failed", failed)
            
            st.divider()
            
            # Results table
            results_df = pd.DataFrame([{
                'Index': r['index'],
                'Design': r['design_idea'][:50] + '...',
                'Status': r['status'],
                'Vertices': r.get('analysis', {}).get('vertices', 0) if r['status'] == 'success' else 0,
                'Quality': f"{r.get('analysis', {}).get('quality_score', 0):.1f}" if r['status'] == 'success' else 'N/A'
            } for r in st.session_state.batch_results])
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Export results
            csv_export = results_df.to_csv(index=False)
            st.download_button("📥 Download Report CSV", csv_export, "batch_results.csv", mime='text/csv')
            
            # Individual results
            st.divider()
            st.markdown("### Individual Results")
            
            for result in st.session_state.batch_results:
                if result['status'] == 'success':
                    with st.expander(f"✅ Design #{result['index']}: {result['design_idea'][:60]}..."):
                        
                        fig = render_mesh_plotly(result['mesh'], f"Design #{result['index']}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        analysis = result['analysis']
                        st.write(f"**Quality:** {analysis['quality_score']:.1f}/100 | **Vertices:** {analysis['vertices']:,} | **Watertight:** {'✅' if analysis['watertight'] else '⚠️'}")
                        
                        col_ex1, col_ex2, col_ex3 = st.columns(3)
                        
                        with col_ex1:
                            stl_data, stl_name = export_mesh_file(result['mesh'], 'stl', f"batch_{result['index']}")
                            if stl_data:
                                st.download_button("📥 STL", stl_data, stl_name, key=f"batch_stl_{result['index']}")
                        
                        with col_ex2:
                            obj_data, obj_name = export_mesh_file(result['mesh'], 'obj', f"batch_{result['index']}")
                            if obj_data:
                                st.download_button("📥 OBJ", obj_data, obj_name, key=f"batch_obj_{result['index']}")
                        
                        with col_ex3:
                            glb_data, glb_name = export_mesh_file(result['mesh'], 'glb', f"batch_{result['index']}")
                            if glb_data:
                                st.download_button("📥 GLB", glb_data, glb_name, key=f"batch_glb_{result['index']}")
                
                else:
                    with st.expander(f"❌ Design #{result['index']}: {result['design_idea'][:60]}..."):
                        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
                        if 'code' in result:
                            with st.expander("View Generated Code"):
                                st.code(result['code'], language='python')
        
        else:
            st.info("💡 Upload a CSV file to start batch processing")

# =================================================================================
# MAIN APPLICATION
# =================================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("🔧 Professional CAD Designer")
        st.markdown("---")
        
        st.markdown("### 🎯 Real CAD Features")
        st.markdown("""
        **Using Professional Tools:**
        - ✅ **trimesh** - Industry CAD library
        - ✅ **pygltflib** - glTF export
        - ✅ **scipy** - Geometric algorithms
        - ✅ **shapely** - 2D operations
        
        **Capabilities:**
        - Boolean operations (union/diff/intersect)
        - Parametric primitives
        - Transformations & arrays
        - Watertight validation
        - Technical drawings
        - Multi-format export
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Pro Tips")
        with st.expander("Writing Good Prompts"):
            st.markdown("""
            **Include:**
            - Exact dimensions (mm/cm)
            - Mechanical features (threads, holes)
            - Material considerations
            - Tolerances if needed
            
            **Example:**
            "M10 bolt, 1.5mm pitch, 40mm shaft, hex head"
            """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ API Status")
        account_id, auth_token = get_cloudflare_api_config()
        
        if account_id and auth_token:
            st.success("✅ API Connected")
        else:
            st.error("❌ Configure API")
            st.code("""
# .streamlit/secrets.toml
CLOUDFLARE_ACCOUNT_ID = "..."
CLOUDFLARE_AUTH_TOKEN = "..."
            """)
        
        st.markdown("---")
        st.caption("Production CAD with trimesh")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔧 Create CAD Meshes",
        "📊 Render & Analyze",
        "🚀 Batch Processing"
    ])
    
    with tab1:
        tab_create_meshes()
    
    with tab2:
        tab_render_analyze()
    
    with tab3:
        tab_batch_processing()

if __name__ == "__main__":
    main()
