import streamlit as st
import google.generativeai as genai
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
        st.write("AI Response:", model_description)
        return model_description
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None

# Function to extract parameters (dimensions) from the generated mesh code
def extract_mesh_parameters(model_description):
    # Extract vertices and faces information from the description
    vertices = []
    faces = []
    
    # Regex patterns to extract data from Gemini's output (assumed format)
    vertex_pattern = re.findall(r'Vertex\[(.*?)\]', model_description)
    face_pattern = re.findall(r'Face\[(.*?)\]', model_description)
    
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
    
    return vertices, faces

# Function to visualize 3D model using Plotly
def visualize_3d_model(vertices, faces):
    if vertices.ndim == 1:  # If vertices is 1D, reshape to 2D
        vertices = vertices.reshape(-1, 3)

    # Ensure faces is properly shaped as well
    if faces.ndim == 1:  # If faces is 1D, reshape to 2D
        faces = faces.reshape(-1, 3)

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
            
            # Step 3: Visualize the 3D model using Plotly
            if vertices is not None and faces is not None:
                visualize_3d_model(vertices, faces)
            else:
                st.error("Error: Invalid mesh data returned by AI.")
    else:
        st.error("Please enter a description for the 3D model.")
