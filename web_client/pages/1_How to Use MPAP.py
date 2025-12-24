import streamlit as st
from pathlib import Path

# Get the web_client directory (parent of pages/)
# Script location: web_client/pages/1_How to Use MPAP.py
# Images location: web_client/fig*.png (one level up from pages/)
_web_client_dir = Path(__file__).parent.parent.resolve()

# Helper function to get image path
def get_image_path(filename):
    """Get path to image file in web_client directory (one level up from pages/)."""
    img_path = _web_client_dir / filename
    # Use absolute path to ensure it works regardless of working directory
    return str(img_path.resolve())


st.markdown("# How to Use MPAP❓")


st.markdown(     """#####  1. Select microplastics type.    """)

st.write('<p style="color:blue;">In this model, there are six types of microplastics available for selection. Choose one target microplastic.</p>',
         unsafe_allow_html=True)
st.write("")
st.image(get_image_path("fig2.png"))
st.write("")
st.markdown( """##### 2. Input microplastics average size (µm).""")
st.write('<p style="color:blue;">Please enter the average particle size of the target microplastic, in micrometers (µm).</p>',
         unsafe_allow_html=True)

st.markdown( """##### 3. Select water type.""")
st.image(get_image_path("fig3.png"))
st.write("")
st.markdown( """##### 4. Input Organic Pollutants SMILES.""")
st.write('<p style="color:blue;">Please enter the SMILES of the organic pollutants.</p>',unsafe_allow_html=True)
st.write('<p style="color:blue;">**SMILES (Simplified Molecular Input Line Entry System)** is a specification that uses ASCII strings to describe molecular structures. It can unambiguously represent atomic composition, bond types, ring structures, branching, chirality, aromaticity, and other molecular features.</p>',unsafe_allow_html=True)
st.write('<p style="color:blue;">You can convert molecular structure files (e.g., `.mol`, `.sdf`) into SMILES using open-source cheminformatics libraries (such as **RDKit**). Alternatively, if the compound name or **CAS number** is known, its SMILES notation can be retrieved via open-source tools.</p>',unsafe_allow_html=True)
st.markdown( """##### 5. Click the Run Prediction button.""")
st.image(get_image_path("fig4.png"))
st.write('<p style="color:red;">The figure is an example of prediction result.</p>',
         unsafe_allow_html=True)
st.write('<p style="color:red;">If you need to batch process the dataset, please contact us: yangxihe@zju.edu.cn  / xianliu@rcees.ac.cn</p>',unsafe_allow_html=True)





