import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Set up page configuration
st.set_page_config(
    page_title="Adult Census Income",
    layout="wide",
)

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS for styling
load_css("D:\luminar\proj1\mlprj1\styles.css")

# Header with clickable title
st.markdown(
    "<div class='header'><a href='https://example.com' target='_blank'>Adult Census Income</a></div>",
    unsafe_allow_html=True
)

# Container with two boxes for About and Project
st.markdown("<div class='container'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class='box'>
            <h2>About</h2>
            <p>Learn about the project, the dataset, and the problem we aim to solve.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("About"):
        switch_page("About")  # Navigate to the "About" page

with col2:
    st.markdown("""
        <div class='box'>
            <h2>Project</h2>
            <p>Discover the machine learning models and their implementation.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Project"):
        switch_page("Project")  # Navigate to the "Project" page

st.markdown("</div>", unsafe_allow_html=True)

# Footer with LinkedIn link
st.markdown("""
<footer>
    <p>Created by Afsal Salim - <a href="https://www.linkedin.com/in/afsal-salim-97b880288" style="color: #00ffcc;">LinkedIn</a></p>
</footer>
""", unsafe_allow_html=True)
