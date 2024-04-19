import streamlit as st
from multiapp import MultiApp
import random
import time
import warnings
warnings.filterwarnings('ignore') 
import Inference # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Font Classification", Inference.app)

# The main app
app.run()