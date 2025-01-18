import streamlit as st
import google.generativeai as genai
import cadquery as cq
from io import BytesIO
import json

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("AI CAD Co-Pilot for Hardware Design")
st.write("Transform plain text descriptions into parametric 3D models like AutoCAD, Fusion 360, or SolidWorks.")

# Input field for text description
description = st.text_area("Describe your 3D model (e.g., 'A 20mm diameter cylindrical part with a 5mm hole at the center')")

# Button to generate model
if st.button("Generate Model"):
    try:
        # Step 1: Use AI model to interpret the description and extract parameters
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"Generate parameters for a parametric 3D model based on this description: {description}"
        
        # Generate response from the AI model
        response = model.generate_content(prompt)

        # Assuming the AI response contains JSON-like structure of parameters
        model_parameters = json.loads(response.text)  # The AI should output a structured JSON (e.g., {"type": "cylinder", "radius": 10, "height": 20})

        st.write("AI generated parameters:", model_parameters)

        # Step 2: Convert the parameters into a parametric 3D model (using CadQuery or similar)
        
        # Base Shape Creation Example (Additional operations based on the input description)
        if model_parameters["type"] == "cylinder":
            radius = model_parameters.get("radius", 10)  # Default to 10mm if no radius is specified
            height = model_parameters.get("height", 20)  # Default to 20mm if no height is specified
            hole_radius = model_parameters.get("hole_radius", 5)  # Optional hole radius

            result = cq.Workplane("XY").circle(radius).extrude(height)
            if hole_radius:
                result = result.faces(">Z").hole(hole_radius)
        
        elif model_parameters["type"] == "box":
            length = model_parameters.get("length", 10)
            width = model_parameters.get("width", 10)
            height = model_parameters.get("height", 10)
            result = cq.Workplane("XY").box(length, width, height)
        
        # Example for adding additional features:
        elif model_parameters["feature"] == "fillet":
            radius = model_parameters.get("radius", 2)
            result = result.edges().fillet(radius)

        elif model_parameters["feature"] == "chamfer":
            distance = model_parameters.get("distance", 1)
            angle = model_parameters.get("angle", 45)
            result = result.edges().chamfer(distance, angle)

        elif model_parameters["feature"] == "hole":
            hole_type = model_parameters.get("hole_type", "simple")
            diameter = model_parameters.get("diameter", 5)
            result = result.faces(">Z").hole(diameter)

        elif model_parameters["feature"] == "extrude_cut":
            # Assuming a 2D sketch and cutting the shape from the object
            result = result.faces(">Z").cut(result)

        # Additional Example for Extrusion, Revolve, Sweep, etc.:
        if model_parameters["operation"] == "extrude":
            sketch = model_parameters.get("sketch", "circle")
            if sketch == "circle":
                result = cq.Workplane("XY").circle(model_parameters.get("radius", 10)).extrude(model_parameters.get("height", 20))
        
        # More features like pattern, mirror, combine, etc. can be added similarly.
        
        # Step 3: Export model to STL
        stl_bytes = result.toStl()

        # Save STL to file and offer for download
        stl_filename = "generated_model.stl"
        st.download_button(
            label="Download STL Model",
            data=stl_bytes,
            file_name=stl_filename,
            mime="application/stl"
        )

        # Step 4: Optionally, visualize the 3D model (using `pythreejs` or other libraries)
        st.write("Preview the generated model (optional):")
        # (Optional) Displaying a 3D visualization using Plotly or PyThreeJS.
        # Here, you would integrate Plotly or another 3D library for real-time model visualization.

        # For simplicity, display dimensions
        st.write(f"Model Type: {model_parameters['type']}")
        st.write(f"Dimensions: {model_parameters.get('radius', 'N/A')} mm radius, {model_parameters.get('height', 'N/A')} mm height")

    except Exception as e:
        st.error(f"Error: {e}")
