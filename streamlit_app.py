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
        # Requesting model description from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        model_description = response.text
        
        # Log the full AI response to understand its structure
        st.write("Full AI Response:\n", model_description)  # This will log the full text output
        
        return model_description
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None

# Function to extract parameters (dimensions) from the generated mesh code
def extract_mesh_parameters(model_description):
    vertices = []
    faces = []
    
    # Log the raw response to help debug
    st.write("Raw AI Response for Parsing:", model_description)
    
    # Adjust these regular expressions based on actual format
    vertex_pattern = re.findall(r'points?\s*[:]\s*\[(.*?)\]', model_description)
    face_pattern = re.findall(r'polygons?\s*[:]\s*\[(.*?)\]', model_description)

    # Debugging: Print extracted patterns to verify if they match the expected format
    st.write("Extracted Vertex Pattern:", vertex_pattern)
    st.write("Extracted Face Pattern:", face_pattern)
    
    # Parsing vertices (ensure each vertex is a 3D coordinate)
    for vertex in vertex_pattern:
        vertex_values = list(map(float, vertex.split(',')))
        if len(vertex_values) == 3:  # Ensure it has 3 coordinates
            vertices.append(vertex_values)
    
    # Parsing faces (ensure each face is a triplet of indices)
    for face in face_pattern:
        face_indices = list(map(int, face.split(',')))
        if len(face_indices) == 3:  # Ensure it's a triangular face
            faces.append(face_indices)

    # Convert vertices and faces into numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Debugging: Print arrays to verify shape
    st.write("Vertices Array:", vertices)
    st.write("Faces Array:", faces)
    
    # Check if the arrays are non-empty and have the expected shape
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        st.error("Error: No valid vertices or faces were extracted from the AI response.")
        return None, None

    return vertices, faces

# Function to create a Trimesh object from vertices and faces
def create_trimesh_from_parameters(vertices, faces):
    # Check if vertices and faces are valid (non-empty)
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        st.error("Error: Invalid mesh data. Cannot create mesh.")
        return None
    
    # Create a Trimesh object from vertices and faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh

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

                # Step 4: Visualize the 3D model using Plotly
                if mesh:
                    visualize_3d_model(vertices, faces)
                    
                    # Optionally, you can display the mesh in Streamlit for download or further manipulation
                    st.write("Trimesh object created successfully!")
                    st.write(mesh)
                    
                    # Save the mesh to a file for download (optional)
                    try:
                        mesh.export('generated_model.stl')
                        st.download_button("Download STL", 'generated_model.stl')
                    except Exception as e:
                        st.error(f"Error exporting the model: {e}")
                else:
                    st.error("Error: Mesh creation failed.")
            else:
                st.error("Error: Invalid mesh data returned by AI.")
    else:
        st.error("Please enter a description for the 3D model.")
