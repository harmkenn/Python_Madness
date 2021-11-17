import streamlit as st
from multiapp import MultiApp
from apps import a_allgames,b_brackets,c_seedhistory,d_teamwins,e_teamrank,f_backpredict,g_fullpredict,h_elasticnet,i_bracketmaker  # import your app modules here
st.set_page_config(layout="wide")
app = MultiApp()

# Add all your application here
app.add_app("All Games", a_allgames.app)
app.add_app("All Brackets", b_brackets.app)
app.add_app("Seed History", c_seedhistory.app)
app.add_app("Team Wins", d_teamwins.app)
app.add_app("Team Rank", e_teamrank.app)
app.add_app("Back Predict", f_backpredict.app)
app.add_app("Full Predict", g_fullpredict.app)
app.add_app("Elastic Net", h_elasticnet.app)
app.add_app("Bracket Maker", i_bracketmaker.app)

# The main app
app.run()