import os
import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output, State

from rag import Rag

all_config = {
    "OUI": {
        "title": "האוניברסיטה הפתוחה - ייעוץ כללי",
        "description": "אני בוט הייעוץ הכללי של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים באוניברסיטה הפתוחה שאינן ספציפיות לתוכנית לימודים כזו או אחרת."
    },
    "CS": {
        "title": "האוניברסיטה הפתוחה - ייעוץ למדעי המחשב",
        "description": "אני בוט הייעוץ של הפקולטה למדעי המחשב של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים בפקולטה."
    }
}

##############################################################################
##############################################################################
##############################################################################
class DashChat:
    ##############################################################################
    def __init__(self, rag_obj):
        self.rag_obj = rag_obj
        self.welcome_msg = "שלום, איך אוכל לעזור לך היום?"
        self.user_name = "user"
        self.bot_name = "bot"
        self.config = {}
        self.app = dash.Dash(__name__)
        # Remove the setup_callbacks() call from here

    ##############################################################################
    def _init_layout(self, title, subtitle):
        return html.Div([
            # Title for the chat app (loaded from configuration)
            html.H1(title, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl'}),
            html.H3(subtitle, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl'}),
            
            # Chat history with round edges, scrollable area and welcome message
            html.Div(id='chat-history', children=[
                html.Div(self.welcome_msg, style={
                    'text-align': 'right', 
                    'direction': 'rtl', 
                    'color': 'green', 
                    'padding': '10px', 
                    'background-color': '#e8f5e9', 
                    'border-radius': '10px', 
                    'margin': '5px 0'
                })
            ], style={
                #'flex': '1',  # Allow the chat history to grow and shrink
                'height': '400px', 
                'overflow-y': 'auto', 
                'border': '1px solid #ccc', 
                'padding': '10px', 
                'border-radius': '10px', 
                'background-color': '#f9f9f9',
                'margin-bottom': '10px',
                'resize': 'vertical',  # Allow resizing vertically
                'overflow': 'auto'  # Ensure content is scrollable when resized
            }),

            # Container for the input field and send button with RTL direction
            html.Div([
                # Send button
                html.Button('שלח', id='send-button', n_clicks=0, style={
                    'width': '10%', 
                    'padding': '10px', 
                    'border-radius': '10px', 
                    'border': '1px solid #ccc', 
                    'background-color': '#4CAF50', 
                    'color': 'white', 
                    'font-size': '16px',
                    'margin-right': '10px'  # Add space between button and input
                }),

                # User input field
                dcc.Input(id='user-input', type='text', placeholder='הקלד הודעה...', style={
                    'width': '85%', 
                    'padding': '10px', 
                    'border-radius': '10px', 
                    'border': '1px solid #ccc'
                })
            ], style={'display': 'flex', 'flex-direction': 'row-reverse', 'direction': 'rtl', 'justify-content': 'flex-end'}),


            # Hidden div to store chat messages
            html.Div(id='hidden-chat-store', style={'display': 'none'})
        ])

    ##############################################################################
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('chat-history', 'children'),
             Output('hidden-chat-store', 'children'),
             Output('user-input', 'value')],
            [Input('send-button', 'n_clicks'),
             Input('chat-history', 'children')],
            [State('user-input', 'value'),
             State('hidden-chat-store', 'children')],
            prevent_initial_call=True
        )
        def update_chat(n_clicks, chat_history_children, user_message, chat_history):
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if triggered_id == 'send-button' and user_message:
                print("User Message:", user_message[::-1])
                if chat_history is None:
                    chat_history = [{'sender': self.bot_name, 'message': self.welcome_msg}]
                else:
                    chat_history = chat_history.replace("\'", "\"")
                    chat_history = json.loads(chat_history)

                bot_response = self.rag_obj.get_rag_response(user_message, chat_history)

                full_query = chat_history
                full_query.append({'sender': self.user_name, 'message': user_message})
                full_query.append({'sender': self.bot_name, 'message': bot_response})

                chat_layout = []
                for entry in full_query:
                    style = {
                        'text-align': 'right', 
                        'direction': 'rtl', 
                        'padding': '10px', 
                        'border-radius': '10px', 
                        'margin': '5px 0'
                    }
                    if entry['sender'] == self.bot_name:
                        style.update({'color': 'green', 'background-color': '#e8f5e9'})
                    else:
                        style.update({'color': 'blue', 'background-color': '#e0f7fa'})
                    
                    chat_layout.append(html.Div(children=[dcc.Markdown(entry['message'], style=style)]))

                return chat_layout, json.dumps(full_query), ''  # Clear input field
            
            elif triggered_id == 'chat-history':
                # Scroll to bottom logic
                return dash.no_update, dash.no_update, dash.no_update

            return dash.no_update, dash.no_update, dash.no_update

    ##############################################################################
    def init(self, title, subtitle):
        self.app.layout = self._init_layout(title, subtitle)

    ##############################################################################
    def run(self, port):
        self._setup_callbacks()  # Move this line after setting the layout
        self.app.run_server(port=port, debug=True)

##############################################################################
def main(faculty_code):
    # Fetch the port from the environment
    port = int(os.getenv('PORT', 8050))
    config = all_config[faculty_code]

    rag = Rag()
    rag.init(faculty_code)

    dash_chat = DashChat(rag)
    title = config["title"]
    subtitle = config["description"]
    dash_chat.init(title, subtitle)

    dash_chat.run(port)

if __name__ == "__main__":
    main("CS")