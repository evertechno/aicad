import streamlit as st
import google.generativeai as genai
import trimesh
import plotly.graph_objects as go
import numpy as np
import re

# Configure the Gemini API key securely from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit app UI
st.title("AI-Powered CAD Tool")
st.write("Generate 3D models from text descriptions. Enter the description and let AI generate the mesh code!")

# User input for CAD model description
prompt = st.text_area("Enter your 3D model description (e.g., 'Create a toy car with length 100mm, width 50mm, height 30mm'):")

# Function to generate mesh code from text description using Gemini
def generate_mesh_code(prompt):
    try:
        if not prompt:
            st.error("Prompt cannot be empty.")
            return None
        # Requesting model description from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        model_description = response.text
        st.write("AI Response:", model_description)
        return model_description
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None

# Function to extract parameters (dimensions) from the generated mesh code
def extract_mesh_parameters(model_description):
    vertices = []
    faces = []
    
    # Regex patterns to extract data from Gemini's output (assumed format)
    vertex_pattern = re.findall(r'Vertex

\[(.*?)\]

', model_description)
    face_pattern = re.findall(r'Face

\[(.*?)\]

', model_description)

    # Debugging: Print extracted patterns to verify if they match the expected format
    st.write("Extracted Vertex Pattern:", vertex_pattern)
    st.write("Extracted Face Pattern:", face_pattern)
    
    # Parsing vertices
    for vertex in vertex_pattern:
        vertex_values = list(map(float, vertex.split(',')))
        vertices.append(vertex_values)
    
    # Parsing faces (triangular faces)
    for face in face_pattern:
        face_indices = list(map(int, face.split(',')))
        faces.append(face_indices)

    # Convert vertices and faces into numpy arrays (ensure the correct shape)
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Debugging: Print arrays to ensure they have the correct shape
    st.write("Vertices Array:", vertices)
    st.write("Faces Array:", faces)
    
    return vertices, faces

# Function to create a Trimesh object from vertices and faces
def create_trimesh_from_parameters(vertices, faces):
    try:
        # Ensure vertices and faces are not empty
        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Vertices or faces array is empty.")
        
        # Ensure the shapes are correct
        if vertices.ndim != 2 or faces.ndim != 2:
            raise ValueError("Vertices or faces array has incorrect dimensions.")
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    except Exception as e:
        st.error(f"Error creating Trimesh object: {e}")
        return None

# Function to visualize 3D model using Plotly
def visualize_3d_model(vertices, faces):
    if vertices.ndim == 1:  # If vertices is 1D, reshape to 2D
        vertices = vertices.reshape(-1, 3)
    if faces.ndim == 1:  # If faces is 1D, reshape to 2D
        faces = faces.reshape(-1, 3)

    # Debugging: Check the shape before passing to Plotly
    st.write("Vertices Shape:", vertices.shape)
    st.write("Faces Shape:", faces.shape)

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.5,
        color='blue'
    )])
    fig.update_layout(
        title="Generated 3D Model",
        scene=dict(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False)
        ),
    )
    st.plotly_chart(fig)

# Function to tweak parameters and regenerate the model
def tweak_parameters():
    st.subheader("Tweak Model Parameters")
    x_scale = st.slider("X Scale", 0.5, 2.0, 1.0)
    y_scale = st.slider("Y Scale", 0.5, 2.0, 1.0)
    z_scale = st.slider("Z Scale", 0.5, 2.0, 1.0)
    
    if st.button("Apply Tweaks"):
        vertices *= np.array([x_scale, y_scale, z_scale])
        visualize_3d_model(vertices, faces)

# Button to generate 3D model from Gemini
if st.button("Generate 3D Model"):
    if prompt:
        # Step 1: Use Gemini AI to generate the mesh code from the description
        model_description = generate_mesh_code(prompt)
        
        # Step 2: Extract vertices and faces from the mesh code returned by Gemini
        if model_description:
            vertices, faces = extract_mesh_parameters(model_description)
            
            # Step 3: Create Trimesh object from the vertices and faces
            if vertices is not None and faces is not None:
                mesh = create_trimesh_from_parameters(vertices, faces)
                if mesh is not None:
                    # Step 4: Visualize the 3D model using Plotly
                    visualize_3d_model(vertices, faces)
                    
                    # Optionally, you can display the mesh in Streamlit for download or further manipulation
                    st.write("Trimesh object created successfully!")
                    st.write(mesh)
                    
                    # Save the mesh to a file for download (optional)
                    try:
                        mesh.export('generated_model.stl')
                        st.download_button("Download STL", 'generated_model.stl')
                        mesh.export('generated_model.obj')
                        st.download_button("Download OBJ", 'generated_model.obj')
                    except Exception as e:
                        st.error(f"Error exporting mesh: {e}")

                    # Allow tweaking of parameters
                    tweak_parameters()
                else:
                    st.error("Error: Failed to create Trimesh object.")
            else:
                st.error("Error: Invalid mesh data returned by AI.")
    else:
        st.error("Please enter a description for the 3D model.")

# Reset button to clear the form and restart
if st.button("Reset"):
    st.experimental_rerun()
