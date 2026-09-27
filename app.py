"""Interface Volley Analytics : python -m streamlit run app.py"""
import streamlit as st

from views.common import PAGES, inject_css

st.set_page_config(page_title="Volley Analytics", page_icon=":material/sports_volleyball:", layout="wide")
inject_css()

page = st.navigation({
    "Analyses": [
        st.Page(PAGES["dashboard"], title="Tableau de bord", icon=":material/monitoring:", default=True),
        st.Page(PAGES["upload"], title="Nouvelle analyse", icon=":material/upload:", url_path="nouvelle-analyse"),
    ],
    "Aide": [st.Page(PAGES["guide"], title="Guide et lexique", icon=":material/menu_book:", url_path="guide")],
})
page.run()
