import streamlit as st
import google.generativeai as genai
import trimesh
import plotly.graph_objects as go
import re

# Configure the Gemini API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit app UI
st.title("AI-Powered CAD Co-Pilot for Hardware Design")
st.write("Transform your plain text descriptions into parametric 3D models.")

# User input for CAD model description
prompt = st.text_area("Enter your 3D model description (e.g., 'create a toy car with length 10mm, width 5mm, and height 15mm'):")

# Function to generate 3D model from text description using Google Gemini
def generate_3d_model(prompt):
    try:
        # Using Gemini to interpret the prompt and return model parameters
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        model_description = response.text
        st.write("AI Response:", model_description)
        return model_description
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None

# Function to extract parameters from description
def extract_parameters(description):
    # Extract dimensions and parts for the toy car (body, wheels, etc.)
    car_dimensions = re.findall(r'length\s?(\d+\.?\d*)\s?mm.*?width\s?(\d+\.?\d*)\s?mm.*?height\s?(\d+\.?\d*)\s?mm', description)
    wheel_size = re.findall(r'wheel\s?radius\s?(\d+\.?\d*)\s?mm', description)

    # If dimensions found, return them
    if car_dimensions:
        length, width, height = map(float, car_dimensions[0])
        wheel_radius = float(wheel_size[0]) if wheel_size else 2
        return {"shape": "toy_car", "body_dimensions": [length, width, height], "wheel_radius": wheel_radius}
    
    # Default toy car if no valid input is found
    return {"shape": "toy_car", "body_dimensions": [10, 5, 15], "wheel_radius": 2}

# Function to create a 3D model of the toy car using Trimesh
def create_3d_model_from_description(description):
    params = extract_parameters(description)

    # Toy car body (simple box)
    body = trimesh.primitives.Box(extents=params["body_dimensions"])

    # Create the wheels as cylinders (defaulting to 2 wheels if not specified)
    wheel_radius = params["wheel_radius"]
    wheel_height = 2  # fixed height for the wheels
    wheels = []
    for i in range(4):  # Assuming 4 wheels
        # Place wheels symmetrically around the body
        offset_x = params["body_dimensions"][0] / 2 if i % 2 == 0 else -params["body_dimensions"][0] / 2
        offset_y = params["body_dimensions"][1] / 2 if i < 2 else -params["body_dimensions"][1] / 2
        wheel = trimesh.primitives.Cylinder(radius=wheel_radius, height=wheel_height)
        wheel.apply_translation([offset_x, offset_y, -params["body_dimensions"][2] / 2])
        wheels.append(wheel)
    
    # Combine body and wheels
    all_parts = [body] + wheels
    car_model = trimesh.util.concatenate(all_parts)
    return car_model

# Function to visualize the 3D model using Plotly
def visualize_3d_model(model):
    # Convert the trimesh model to vertices and faces for Plotly visualization
    vertices = model.vertices
    faces = model.faces
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

# Button to generate 3D model
if st.button("Generate 3D Model"):
    if prompt:
        # Generate the model description via Gemini
        model_description = generate_3d_model(prompt)
        
        # If a description is generated, proceed to create the 3D model
        if model_description:
            # Generate 3D model from description
            model = create_3d_model_from_description(model_description)

            # Visualize the 3D model using Plotly
            if model:
                visualize_3d_model(model)
    else:
        st.error("Please enter a description for the 3D model.")
