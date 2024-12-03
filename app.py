import streamlit as st
import pdf2image
import time
from PIL import Image
import io
import base64
from pathlib import Path
import tempfile
from datetime import datetime
import requests
import zipfile

# ================================
# Streamlit Page Configuration
# ================================
st.set_page_config(
    page_title="📄 PDF to Text Analyzer",
    layout="wide",
    page_icon="📄"
)

# ================================
# Initialize Session State
# ================================
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'final_text' not in st.session_state:
    st.session_state.final_text = ""

# ================================
# Convert PDF to Images
# ================================
def convert_pdf_to_images(pdf_file):
    """Convert PDF to list of images."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        images = pdf2image.convert_from_path(pdf_path)
        Path(pdf_path).unlink()
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return []

# ================================
# Encode Image to Base64
# ================================
def encode_image(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ================================
# Analyze Single Image using Gemini API
# ================================
def analyze_single_image(encoded_image, page_num, api_key):
    """Analyze a single image using Gemini 1.5 Flash API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Construct the request payload for a single image
    contents = [{
        "role": "user",
        "parts": [
            {"text": f"You are a document text extractor. For page {page_num}, please:\n1. Write out ALL text visible in the image exactly as it appears\n2. Preserve formatting such as bullet points, numbering, and paragraphs\n3. Include any visible headers, footers, and slide numbers\n4. After writing all text, provide a brief content summary"},
            {
                "inlineData": {
                    "data": encoded_image,
                    "mimeType": "image/png"
                }
            }
        ]
    }]
    
    payload = {"contents": contents}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text
        else:
            error_msg = f"Failed to analyze page {page_num}: API returned status code {response.status_code}"
            st.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error analyzing page {page_num}: {str(e)}"
        st.error(error_msg)
        return error_msg

# ================================
# Main Function
# ================================
def main():
    st.title("📄 PDF to Text Analyzer")
    st.markdown("#### Upload a PDF to convert it to text with smart image analysis")
    
    # Sidebar for API Key configuration
    with st.sidebar:
        st.markdown("## 🔑 Google Gemini API Configuration")
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from [Google AI Studio](https://ai.google.com/tools/)"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your Gemini API key to proceed.")
        
        st.markdown("---")
        
        # Option to clear session state
        if st.button("🔄 Clear Session"):
            st.session_state.clear()
            st.experimental_rerun()
    
    # Main content area with tabs
    tabs = st.tabs(["📤 Upload & Process", "📄 Analysis Results", "📦 Download"])
    
    # ================================
    # Upload & Process Tab
    # ================================
    with tabs[0]:
        uploaded_file = st.file_uploader("📂 Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if not st.session_state.api_key:
                st.error("❌ Please configure your Gemini API key in the sidebar first!")
                return
            
            with st.spinner("🔄 Processing PDF..."):
                # Convert PDF to images
                images = convert_pdf_to_images(uploaded_file)
                
                if not images:
                    st.error("❌ Failed to process PDF.")
                    return
                
                # Create columns for organization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Process images one by one
                    analyses = []
                    progress_bar = st.progress(0, text="Starting analysis...")
                    
                    # Process each image
                    for idx, img in enumerate(images, 1):
                        # Update progress
                        progress_bar.progress((idx-1)/(len(images)), 
                                           text=f"Analyzing page {idx}/{len(images)}...")
                        
                        # Convert image to base64
                        encoded_image = encode_image(img)
                        
                        # Analyze single image
                        analysis = analyze_single_image(encoded_image, idx, st.session_state.api_key)
                        analyses.append(analysis)
                    
                    # Combine results into final text
                    final_text = ""
                    for i, analysis in enumerate(analyses, 1):
                        final_text += f"\n### 📄 Page {i}\n"
                        final_text += f"{analysis}\n\n---\n"
                    
                    # Store results and complete
                    st.session_state.final_text = final_text
                    st.session_state.processed_images = images
                    progress_bar.progress(100, text="✅ Processing complete!")
                
                with col2:
                    # Display thumbnails
                    st.markdown("### 📄 Processed Pages")
                    for idx, img in enumerate(images, 1):
                        st.image(img, width=200, caption=f"Page {idx}")
    
    # ================================
    # Analysis Results Tab
    # ================================
    with tabs[1]:
        st.markdown("### 📝 Analysis Result")
        if st.session_state.final_text:
            st.text_area("📄 Analysis Result", st.session_state.final_text, height=600)
        else:
            st.info("No analysis results to display. Please upload and process a PDF first.")
    
    # ================================
    # Download Tab
    # ================================
    with tabs[2]:
        st.markdown("### 📥 Download Results")
        if st.session_state.final_text:
            # Download Analysis Text
            st.download_button(
                label="⬇️ Download Analysis as TXT",
                data=st.session_state.final_text,
                file_name="pdf_analysis.txt",
                mime="text/plain"
            )
            
            # Download Processed Images as ZIP
            if 'processed_images' in st.session_state:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for idx, img in enumerate(st.session_state.processed_images, 1):
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="PNG")
                        img_bytes = img_buffer.getvalue()
                        zip_file.writestr(f"Page_{idx}.png", img_bytes)
                zip_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Download Images as ZIP",
                    data=zip_buffer,
                    file_name="processed_images.zip",
                    mime="application/zip"
                )
        else:
            st.info("No files to download. Please upload and process a PDF first.")

if __name__ == "__main__":
    main()