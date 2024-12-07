import streamlit as st
import importlib.util
import os

# Set the page layout to wide
st.set_page_config(layout="wide", page_title=f"Python Madness 2025")

# Dictionary that maps .py filenames to user-friendly names
sub_app_names = {
    'a_allgames.py': 'All Games',
    'b_brackets.py': 'All Brackets',
    'c_seedhistory.py': 'Seed History',
    'd_teampase.py': 'Team Performance',
    'd_teamwins.py': 'Team Wins',
    'e_teamrank.py': 'Team Rankings',
    'f_backpredict.py': 'Back Predict',
    'g_fullpredict.py': 'Full Predict',
    'h_update.py': 'Update',
    'i_bracketmaker.py': 'Bracket Maker'
}

# Get a list of .py files from the SubApps folder
sub_apps_folder = os.path.join(os.path.dirname(__file__), 'apps')
sub_apps = [f for f in os.listdir(sub_apps_folder) if f.endswith('.py')]
st.sidebar.title("Python Madness v2025.1")
st.sidebar.subheader("by Ken Harmon")
# Create radio buttons in the sidebar using the user-friendly names
selected_sub_app_name = st.sidebar.radio('', list(sub_app_names.values()))

# Get the corresponding .py filename from the selected name
selected_sub_app = [k for k, v in sub_app_names.items() if v == selected_sub_app_name][0]

# Import and run the selected sub-app
if selected_sub_app:
    spec = importlib.util.spec_from_file_location(selected_sub_app, os.path.join(sub_apps_folder, selected_sub_app))
    sub_app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub_app_module)