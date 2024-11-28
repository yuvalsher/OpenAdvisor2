import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output, State
from AbstractLlm import AbstractLlm
from flask import request

##############################################################################
##############################################################################
##############################################################################
class DashChat:
    ##############################################################################
    def __init__(self, llm_obj: AbstractLlm):
        self.llm_obj = llm_obj
        self.config = {}
        self.app = dash.Dash(__name__)
        self.chat_histories = {}
        self.client_sessions = {}

    ##############################################################################
    def _init_layout(self, title, subtitle):
        return html.Div([
            # Title for the chat app (loaded from configuration)
            html.H1(title, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl'}),
            html.H3(subtitle, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl', 'whiteSpace': 'pre-line'}),
            
            # Chat history with round edges, scrollable area and welcome message
            html.Div(id='chat-history', children=[], style={
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

            # Add hidden div to store client ID
            dcc.Store(id='client-id-store', storage_type='session'),

            # Hidden div to store chat messages
            html.Div(id='hidden-chat-store', style={'display': 'none'})
        ])

    ##############################################################################
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('chat-history', 'children'),
             Output('hidden-chat-store', 'children'),
             Output('user-input', 'value'),
             Output('client-id-store', 'data')],
            [Input('send-button', 'n_clicks'),
             Input('chat-history', 'children')],
            [State('user-input', 'value'),
             State('hidden-chat-store', 'children'),
             State('client-id-store', 'data')],
            prevent_initial_call=True
        )
        def update_chat(n_clicks, chat_history_children, user_message, chat_history_json, client_id):
            msg_sender = self.config["msg_sender_field"]
            msg_text = self.config["msg_text_field"]
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if triggered_id == 'send-button' and user_message:
                # Get the client's chat history
                chat_history = self._get_or_create_chat_history(client_id)

                if user_message == "reset" or user_message == "נקה":
                    print("Resetting chat history")
                    self.chat_histories[client_id] = [{
                        msg_sender: self.bot_name,
                        msg_text: self.welcome_msg
                    }]
                    self.llm_obj.reset_chat_history(client_id)
                    return [], [], '', client_id

                print("User Message:", user_message[::-1])

                # Get the response from the LLM object with client_id
                bot_response, client_id = self.llm_obj.do_query(user_message, chat_history, client_id)

                # Add the user message and the bot response to the chat history
                chat_history.append({msg_sender: self.user_name, msg_text: user_message})
                chat_history.append({msg_sender: self.bot_name, msg_text: bot_response})

                chat_layout = []
                for entry in chat_history:
                    style = {
                        'text-align': 'right', 
                        'direction': 'rtl', 
                        'padding': '10px', 
                        'border-radius': '10px', 
                        'margin': '5px 0'
                    }
                    if entry[msg_sender] == self.bot_name:
                        style.update({'color': 'green', 'background-color': '#e8f5e9'})
                    else:
                        style.update({'color': 'blue', 'background-color': '#e0f7fa'})
                    
                    chat_layout.append(html.Div(children=[dcc.Markdown(entry[msg_text], style=style)]))

                return chat_layout, json.dumps(chat_history), '', client_id
            
            elif triggered_id == 'chat-history':
                # Scroll to bottom logic
                return dash.no_update, dash.no_update, dash.no_update, client_id

            return dash.no_update, dash.no_update, dash.no_update, client_id

    ##############################################################################
    def _get_or_create_chat_history(self, client_id: str) -> list:
        """Get or create chat history for a client."""
        if client_id not in self.chat_histories:
            self.chat_histories[client_id] = [{
                self.config["msg_sender_field"]: self.bot_name,
                self.config["msg_text_field"]: self.welcome_msg
            }]
        return self.chat_histories[client_id]

    ##############################################################################
    def init(self, title, subtitle, config):
        self.config = config
        self.welcome_msg = config["Chat_Welcome_Message"]
        self.user_name = config["user_name"]
        self.bot_name = config["bot_name"]
        self.app.layout = self._init_layout(title, subtitle)

    ##############################################################################
    def run(self, host='0.0.0.0', port=10000, debug=False):
        self._setup_callbacks()
        self.app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    from OpenAdvisor2 import main
    main("Tools", "CS")