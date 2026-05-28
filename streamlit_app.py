# --------- Initialization ---------
# Library
import os
import tempfile

import pandas as pd
import streamlit as st

import VAD as vad


parent_dir = os.path.dirname(os.path.abspath(__file__))

# Page config for better width
st.set_page_config(layout="wide")


# --------- Functions Setup ---------
def process_i3d_feature():

    uploaded_file = st.session_state.i3d_feature_file

    with st.spinner("Performing video anomaly detection..."):

        # Save the uploaded file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_feature_path = temp_file.name

        try:
            result = vad.VAD(temp_feature_path)
        finally:
            os.remove(temp_feature_path)

        st.session_state.result = result
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.page = "result"

# Define a function to create a DataFrame for anomaly frame ranges for display
def range_table(anomaly_frame_ranges):

    rows = []

    for index, frame_range in enumerate(anomaly_frame_ranges):
        row = {
            "No.": index + 1,
            "Start frame": frame_range[0],
            "End frame": frame_range[1],
        }

        rows.append(row)

    return pd.DataFrame(rows)


# --------- User Interface ---------
# Setting screen status
st.session_state.setdefault("page", "input")
st.session_state.setdefault("result", None)

# Title
st.title("Video Anomaly Detection")


# Side bar
logo_path = os.path.join(parent_dir, "figure/apu_logo.png")
st.sidebar.image(logo_path, width=150)

st.sidebar.divider()

st.sidebar.title("Introduction")
st.sidebar.markdown("""
This is a tool to perform video anomaly detection using extracted I3D features.
Only I3D `.npy` feature files are accepted.
""")

st.sidebar.title("Instruction")
st.sidebar.markdown("""
**1. Upload Feature File**

Upload an extracted I3D feature file in `.npy` format. You may refer the GitHub repo Input directory to select the `.npy` file for testing.

If you need more data, you may download it from here:  
https://drive.google.com/file/d/1xKx4QkB_1QS84ecONUYBsg5M9t3TpncC/view

Data source: benedictstar. (2024). *Joint-VAD* [Data set and code repository]. GitHub. Retrieved May 1, 2026, from https://github.com/benedictstar/Joint-VAD

**2. Execute**

Click “Run VAD” button to begin video anomaly detection.

**3. Video-Level Anomaly result**

The app will display the video-level anomaly probability and whether anomaly is detected.

**4. Frame-Level Anomaly result**

The app will show frame-level anomaly probability and detected anomaly frame ranges.

**5. Restart Process**

Click “Start over” button to restart the process.
""")


# Define markdown
st.markdown(
    """
    <style>

    div.stButton > button {
        background-color: #323232;
        color: white;
    }

    .result_area {
        font-size: 1.1rem;
        line-height: 1.8;
        white-space: pre-wrap !important;
        border: 1px solid #d7d7d7;
        background-color: #f4f4f4;
        padding: 16px;
        margin-bottom: 24px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------- Interaction ---------
# If current state is input
if st.session_state.page == "input":

    # File uploader for I3D feature file
    st.file_uploader(
        "Upload an I3D feature file",
        type=["npy"],
        key="i3d_feature_file",
    )

    # Execute VAD when button is clicked
    if st.button("Run VAD"):
        if st.session_state.i3d_feature_file is not None:
            process_i3d_feature()
            st.rerun()
        else:
            st.warning("No I3D feature file uploaded.")


# If current state is result
elif st.session_state.page == "result":

    # Get VAD result
    result = st.session_state.result

    st.markdown(f"<div class='result_area'>File: {st.session_state.uploaded_file_name}</div>", unsafe_allow_html=True)

    # Display video-level anomaly probability and anomaly status
    probability_col, anomaly_col = st.columns(2)

    with probability_col:
        st.metric("Video anomaly probability", f"{result['video_anomaly_prob']:.4f}")

    with anomaly_col:
        if result["is_video_anomaly"]:
            st.error("Anomaly detected")
        else:
            st.success("No anomaly detected")

    # Display frame-level anomaly probability
    st.subheader("Frame-level anomaly probability")
    frame_prob_df = pd.DataFrame({
        "Anomaly probability": result["frame_anomaly_probs"]
    })
    st.line_chart(frame_prob_df)

    # Display anomaly frame ranges
    if result["anomaly_frame_ranges"]:
        st.subheader("Anomaly ranges")
        st.dataframe(
            range_table(result["anomaly_frame_ranges"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No anomaly frame range detected.")

    if st.button("Start over"):

        to_clear = [
            "i3d_feature_file",
            "result",
            "uploaded_file_name",
        ]

        for key in to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.page = "input"
        st.rerun()
