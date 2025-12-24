import streamlit as st
from pathlib import Path

# Get web_client directory for images
_web_client_dir = Path(__file__).parent.resolve()

# Helper function to get image path
def get_image_path(filename):
    """Get path to image file in web_client directory."""
    img_path = _web_client_dir / filename
    return str(img_path.resolve())

st.sidebar.success("Author：👩‍🎓Xian Liu ")
st.sidebar.success("Author：👨‍🎓Xihe Yang ")

st.write("""
<style>
    .centered-text {
        text-align: center;
    }
</style>

<h1 class="centered-text">Welcome to use MPAP! 👋</h1>

<h3 class="centered-text">Predicting the adsorption capacity of microplastics for organic pollutants 😎</h3>
""", unsafe_allow_html=True)
st.write("")
st.image(get_image_path("fig1.png"))
st.write("")

st.write("""
<style>
    .text-justify {
        text-align: justify;
        text-justify: inter-word;
    }
    .custom-h5 {
        font-weight: lighter; /* Adjust the font weight here */
    }
</style>

<div class="text-justify">
    <h5 class="custom-h5">Microplastics (MPs), prevalent in water bodies, soil, and the atmosphere, pose significant risks to environmental and ecological health. By adsorbing hazardous compounds such as organic pollutants, MPs alter pollutants transport and fate. To address this, we developed a multimodal Siamese neural network named Microplastic Pollutant Adsorption Prediction (MPAP), trained on 1,101 adsorption records covering 403 compounds and six MP types. Unlike previous models that rely on single-feature representations, MPAP leverages a multimodal architecture that integrates molecular fingerprints and graph embeddings to capture chemical structure, along with microplastic morphological features such as polymer type and particle size, as well as water chemistry parameters, enabling a more comprehensive characterization of sorption behavior. The model outperforms baseline models with R² = 0.869 on the validation set and 0.863 on the test set. Experimental validation using batch adsorption experiments with four previously untested pollutants, quantified via liquid chromatography-mass spectrometry (LC-MS) or microwave plasma torch ionization-mass spectrometry (MPT-MS), confirmed strong predictive performance. To support broad application, we provide an open-access web tool (http://mpap.envwind.site:8004/) for rapid, high-throughput prediction across diverse MP-pollutant-water environment scenarios.
</h5>
</div>
""", unsafe_allow_html=True)