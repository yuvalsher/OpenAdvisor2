import os
import dash
import json
from dash import dcc, html
from dash.dependencies import Input, Output, State

from rag import initialize_rag, get_rag_response

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

welcome_msg = "שלום, איך אוכל לעזור לך היום?"
user_name = "user"
bot_name = "bot"

config = {}

# Initialize the Dash app
app = dash.Dash(__name__)

##############################################################################
# App layout with round edges, title, and welcome message
def init_layout(title, subtitle):
    return html.Div([
        # Title for the chat app (loaded from configuration)
        html.H1(title, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl'}),
        html.H3(subtitle, style={'text-align': 'center', 'font-family': 'Arial', 'padding': '10px', 'border-bottom': '2px solid #ccc', 'direction': 'rtl'}),
        
        # Chat history with round edges, scrollable area and welcome message
        html.Div(id='chat-history', children=[
            html.Div(welcome_msg, style={
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

# Client-side callback for scrolling
app.clientside_callback(
    """
    function(children) {
        var chatOutput = document.getElementById('chat-history');
        if (chatOutput) {
            chatOutput.scrollTop = chatOutput.scrollHeight;
        }
        return '';
    }
    """,
    Output('user-input', 'value'),
    Input('chat-history', 'children')
)

##############################################################################
# Callback to update the chat history when the user sends a message
@app.callback(
    [Output('chat-history', 'children'),
     Output('hidden-chat-store', 'children')],
    [Input('send-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('hidden-chat-store', 'children')],
    prevent_initial_call=True
)
def update_chat(n_clicks, user_message, chat_history):
    if user_message:
        print("User Message:", user_message[::-1])
        # Initialize chat history if it doesn't exist
        if chat_history is None:
            chat_history = [{'sender': bot_name, 'message': welcome_msg}]
        else:
            # Replace " with \" in the chat history
            #chat_history = chat_history.replace("\"", "\\\"")
            # Replace "\'" with "'" in the chat history
            chat_history = chat_history.replace("\'", "\"")
            chat_history = json.loads(chat_history)

        # Simulate a bot response (replace with real bot logic)
        bot_response = get_rag_response(user_message, chat_history)

        # Add user message to chat history
        full_query = chat_history
        full_query.append({'sender': user_name, 'message': user_message})

        full_query.append({'sender': bot_name, 'message': bot_response})

        # Build chat history layout with round edges
        chat_layout = []
        for entry in full_query:
            if entry['sender'] == bot_name:
                chat_layout.append(
                    html.Div(children=[
                        dcc.Markdown(entry['message'], style={
                            'text-align': 'right', 
                            'direction': 'rtl', 
                            'color': 'green', 
                            'padding': '10px', 
                            'background-color': '#e8f5e9', 
                            'border-radius': '10px', 
                            'margin': '5px 0'
                        })
                    ])
                )
            else:
                chat_layout.append(
                    html.Div(children=[
                        dcc.Markdown(entry['message'], style={
                            'text-align': 'right', 
                            'direction': 'rtl', 
                            'color': 'blue', 
                            'padding': '10px', 
                            'background-color': '#e0f7fa', 
                            'border-radius': '10px', 
                            'margin': '5px 0'
                        })
                    ])
                )

        # Clear user input field with id='user-input'
        return chat_layout, json.dumps(full_query)  #str(full_query)
    
    return dash.no_update, dash.no_update

##############################################################################
def main(faculty_code):
    global config
    config = all_config[faculty_code]
    # Fetch the port from the environment
    port = int(os.getenv('PORT', 8050))

    initialize_rag(faculty_code)

    title = config["title"]
    subtitle = config["description"]

    app.layout = init_layout(title, subtitle)
   
    #app.run_server(host='0.0.0.0', port=port, debug=True)
    app.run_server(port=port, debug=True)

##############################################################################
if __name__ == "__main__":
    main("CS")


