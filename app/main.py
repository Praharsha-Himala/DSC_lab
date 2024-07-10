import streamlit as st
from streamlit_option_menu import option_menu
import predict, References, Info, Authors, Help

st.set_page_config(
    page_title="Phishing URL detection"
)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title":title,
            "function":function
        })

    def run():
        with st.sidebar:
            app = option_menu(
                menu_title="Menu",
                options=['Home', 'About', 'Authors', 'References', 'Help Us Improve'],
                icons=['house-fill', 'chat-fill', 'person-circle', 'book', 'gear-fill'],
                menu_icon='chat-text-fill'
            )

        if app == 'Home':
            predict.app()

        if app == 'About':
            Info.app()

        if app == 'Authors':
            Authors.app()

        if app == 'References':
            References.app()

        if app == 'Help Us Improve':
            Help.app()
    run()