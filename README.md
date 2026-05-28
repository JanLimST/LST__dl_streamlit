## Data Source
This app expects pre-extracted I3D feature files in .npy format.

The I3D feature files used for testing are based on the following repository:

benedictstar. (2024). *Joint-VAD* [Data set and code repository]. GitHub. Retrieved from https://github.com/benedictstar/Joint-VAD

You may refer the GitHub repo here to select the `.npy` file for testing: https://github.com/JanLimST/LST__dl_streamlit/tree/main/input/SH_Test_ten_crop_i3d
Additional feature files can be downloaded from:  
https://drive.google.com/file/d/1xKx4QkB_1QS84ecONUYBsg5M9t3TpncC/view

## Execution step
Steps to execute the Streamlit program:

1. Ensure Python is installed and pip install is available.
2. Install the uv package manager by running the following command: pip install uv
3. Open a terminal and navigate to the project directory.
4. Run the following command to create the virtual environment and install all required dependencies: uv sync
5. Activate the virtual environment:
- Windows (Command Prompt): .venv\Scripts\activate
- Ubuntu / Linux / macOS: source .venv/bin/activate
6. Launch the Streamlit application: streamlit run streamlit_app.py
7. If this method does not work, a requirements.txt file is provided for manual dependency installation.


