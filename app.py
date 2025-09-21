"""
Streamlit Web Interface for Rain Streak Detection
A user-friendly web application for analyzing rain streaks in images
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import sys
import os

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.rain_detector import detect_rainstreaks, calculate_rain_severity
from utils.image_utils import save_image

# Configure Streamlit page
st.set_page_config(
    page_title="RainMark: Detecting and Quantifying Rain Streaks via Local-Weber Contrast and Pixel Saturation",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-header {
        color: #1f77b4;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🌧️ RainMark: Detecting and Quantifying Rain Streaks via Local-Weber Contrast and Pixel Saturation")
    st.markdown("""
    **Analyze and quantify rain streaks in images using Local-Weber Contrast and Pixel Saturation**
    
    This system implements a sophisticated algorithm that:
    - Detects rain streaks using contrast analysis
    - Measures edge amplification and newly saturated pixels
    - Calculates rain coverage percentage
    - Provides visual overlays highlighting detected rain streaks
    """)
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Detection Parameters")
    st.sidebar.markdown("Adjust these parameters to fine-tune the detection algorithm:")
    
    # Parameter controls
    S = st.sidebar.slider(
        "Subwindow Size (S)",
        min_value=3, max_value=15, value=7, step=2,
        help="Size of the local window for contrast computation. Larger values = smoother detection."
    )
    
    visibility_percent = st.sidebar.slider(
        "Visibility Threshold (%)",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="Minimum contrast threshold for edge detection. Lower values = more sensitive."
    )
    
    bright_thresh = st.sidebar.slider(
        "Brightness Threshold",
        min_value=100, max_value=200, value=150, step=10,
        help="Threshold for detecting saturated (bright) pixels. Higher values = less saturation detection."
    )
    
    # Advanced options
    with st.sidebar.expander("📊 Advanced Options"):
        show_intermediate = st.checkbox("Show intermediate results", value=True)
        download_overlay = st.checkbox("Enable overlay download", value=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload Ground Truth Image")
        gt_file = st.file_uploader(
            "Choose ground truth (clean) image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the clean version of the image without rain"
        )
        
        if gt_file is not None:
            gt_image = Image.open(gt_file)
            st.image(gt_image, caption="Ground Truth Image", use_container_width=True)
            gt_array = np.array(gt_image)
    
    with col2:
        st.header("🌧️ Upload Rainy Image")
        rain_file = st.file_uploader(
            "Choose rainy image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the image containing rain streaks"
        )
        
        if rain_file is not None:
            rain_image = Image.open(rain_file)
            st.image(rain_image, caption="Rainy Image", use_container_width=True)
            rain_array = np.array(rain_image)
    
    # Process images when both are uploaded
    if gt_file is not None and rain_file is not None:
        
        # Validation
        if gt_array.shape != rain_array.shape:
            st.error("⚠️ Error: Images must have the same dimensions!")
            st.info(f"Ground Truth: {gt_array.shape}, Rainy Image: {rain_array.shape}")
            return
        
        # Processing button
        st.markdown("---")
        
        if st.button("🔍 Analyze Rain Streaks", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing images... This may take a moment."):
                try:
                    # Run detection
                    results = detect_rainstreaks(
                        gt_array, rain_array, 
                        S=S, 
                        visibility_percent=visibility_percent, 
                        bright_thresh=bright_thresh,
                        show_plots=False
                    )
                    
                    # Calculate rain severity
                    rain_severity = calculate_rain_severity(results)
                    
                    # Display results
                    display_results(results, rain_severity, show_intermediate, download_overlay)
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.info("Please check your images and try again.")
    
    else:
        # Instructions when no images uploaded
        st.markdown("---")
        st.info("👆 Please upload both ground truth and rainy images to start the analysis.")
        
        # Sample usage info
        with st.expander("💡 How to use this tool"):
            st.markdown("""
            **Step 1:** Upload a ground truth (clean) image
            **Step 2:** Upload the corresponding rainy image
            **Step 3:** Adjust detection parameters in the sidebar if needed
            **Step 4:** Click "Analyze Rain Streaks" to process
            
            **Tips:**
            - Images should be the same size and show the same scene
            - JPEG, PNG formats are supported
            - Larger images may take longer to process
            - Adjust parameters if detection is too sensitive or not sensitive enough
            """)


def display_results(results, rain_severity, show_intermediate, download_overlay):
    """Display the analysis results in a user-friendly format"""
    
    st.markdown("---")
    st.markdown('<div class="result-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Edge Amplification",
            value=f"{results['e1']:.3f}",
            help="Ratio of edge enhancement due to rain. Higher values indicate more pronounced rain effects."
        )
    
    with col2:
        st.metric(
            label="New Saturation",
            value=f"{results['ns1']*100:.1f}%",
            help="Percentage of pixels that became saturated due to rain."
        )
    
    with col3:
        st.metric(
            label="Rain Coverage",
            value=f"{results['percentage_streak_area']:.1f}%",
            help="Percentage of image area covered by detected rain streaks."
        )
    
    with col4:
        st.metric(
            label="Rain Severity",
            value=f"{rain_severity:.2f}",
            help="Overall rain severity score combining all metrics."
        )
    
    # Severity interpretation
    if rain_severity < 1.0:
        severity_level = "🟢 Light"
        severity_color = "green"
    elif rain_severity < 5.0:
        severity_level = "🟡 Moderate"
        severity_color = "orange"
    else:
        severity_level = "🔴 Heavy"
        severity_color = "red"
    
    st.markdown(f"**Rain Intensity:** <span style='color: {severity_color}'>{severity_level}</span>", 
                unsafe_allow_html=True)
    
    # Main result visualization
    st.markdown("### 🎯 Rain Streak Detection Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Detected Rain Mask**")
        fig_mask = px.imshow(results['rain_mask'], color_continuous_scale='gray', aspect='auto')
        fig_mask.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_mask, use_container_width=True)
    
    with col2:
        st.markdown("**✨ Rain Streaks Highlighted**")
        st.image(results['overlay'], caption="Neon green highlights show detected rain streaks", 
                use_container_width=True)
    
    # Intermediate results
    if show_intermediate:
        st.markdown("### 🔬 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Edge Detection Comparison**")
            edge_diff = results['edge_map_rain'].astype(float) - results['edge_map_gt'].astype(float)
            fig_edges = px.imshow(edge_diff, color_continuous_scale='RdBu', aspect='auto',
                                 title="Edge Difference (Red: More edges in rain)")
            fig_edges.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_edges, use_container_width=True)
        
        with col2:
            st.markdown("**Brightness Difference**")
            fig_bright = px.imshow(results['brightness_diff'], color_continuous_scale='viridis', aspect='auto',
                                  title="Brightness Difference (Rain - GT)")
            fig_bright.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bright, use_container_width=True)
        
        # Additional metrics
        with st.expander("📈 Detailed Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Detection Statistics:**")
                st.write(f"• Total pixels: {results['rain_mask'].size:,}")
                st.write(f"• Rain pixels detected: {np.sum(results['rain_mask']):,}")
                st.write(f"• Delta threshold: {results['delta_threshold']:.2f}")
            
            with col2:
                st.markdown("**Edge Statistics:**")
                st.write(f"• Edges in GT: {np.sum(results['edge_map_gt']):,}")
                st.write(f"• Edges in Rain: {np.sum(results['edge_map_rain']):,}")
                st.write(f"• Edge increase: {np.sum(results['edge_map_rain']) - np.sum(results['edge_map_gt']):,}")
    
    # Download options
    if download_overlay:
        st.markdown("### 💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert overlay to bytes for download
            overlay_pil = Image.fromarray(results['overlay'])
            overlay_bytes = io.BytesIO()
            overlay_pil.save(overlay_bytes, format='PNG')
            overlay_bytes = overlay_bytes.getvalue()
            
            st.download_button(
                label="📥 Download Highlighted Image",
                data=overlay_bytes,
                file_name="rain_detection_overlay.png",
                mime="image/png"
            )
        
        with col2:
            # Convert mask to bytes for download
            mask_pil = Image.fromarray((results['rain_mask'] * 255).astype(np.uint8))
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format='PNG')
            mask_bytes = mask_bytes.getvalue()
            
            st.download_button(
                label="📥 Download Rain Mask",
                data=mask_bytes,
                file_name="rain_detection_mask.png",
                mime="image/png"
            )


if __name__ == "__main__":
    main()
