import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output, State
from AbstractLlm import AbstractLlm

##############################################################################
##############################################################################
##############################################################################
class DashChat:
    ##############################################################################
    def __init__(self, llm_obj: AbstractLlm):
        self.llm_obj = llm_obj
        self.user_name = "user"
        self.bot_name = "assistant" #"bot"
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
            msg_sender = 'sender'
            msg_text = 'message'
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if triggered_id == 'send-button' and user_message:
                print("User Message:", user_message[::-1])
                if chat_history is None:
                    chat_history = [{msg_sender: self.bot_name, msg_text: self.welcome_msg}]
                else:
                    chat_history = chat_history.replace("\'", "\"")
                    chat_history = json.loads(chat_history)

                # Get the response from the LLM object
                bot_response = self.llm_obj.do_query(user_message, chat_history)

                # Add the user message and the bot response to the chat history
                full_query = chat_history
                full_query.append({msg_sender: self.user_name, msg_text: user_message})
                full_query.append({msg_sender: self.bot_name, msg_text: bot_response})

                chat_layout = []
                for entry in full_query:
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

                return chat_layout, json.dumps(full_query), ''  # Clear input field
            
            elif triggered_id == 'chat-history':
                # Scroll to bottom logic
                return dash.no_update, dash.no_update, dash.no_update

            return dash.no_update, dash.no_update, dash.no_update

    ##############################################################################
    def init(self, title, subtitle, config):
        self.config = config
        self.welcome_msg = config["Chat_Welcome_Message"]
        self.app.layout = self._init_layout(title, subtitle)

    ##############################################################################
    def run(self, host='0.0.0.0', port=10000, debug=False):
        self._setup_callbacks()
        self.app.run_server(host=host, port=port, debug=debug)
