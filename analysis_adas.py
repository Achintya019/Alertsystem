import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import base64
import numpy as np

color_map = {
    'O': None,  # No color for 'O'
    'G': 'green',
    'Y': 'yellow',
    'R': 'red'
}
file_path="/home/achintya-trn0175/Desktop/alertsystem/22June/22.csv"
df=pd.read_csv(file_path)

# Create the Dash app
app = Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Graph(id='speed-plot', config={'displayModeBar': True}),
    html.Div([
        html.Img(id='image-display', style={'width': '50%'}),
        html.Button('Correct Alert', id='correct-button', n_clicks=0),
        html.Button('Incorrect Alert', id='incorrect-button', n_clicks=0),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),
    html.Div(id='dummy-output', style={'display': 'none'}),
    dcc.ConfirmDialog(id='notification-dialog')
])

@app.callback(
    Output('speed-plot', 'figure'),
    Input('speed-plot', 'clickData'),
    State('speed-plot', 'relayoutData')
)
def update_graph(clickData, relayoutData):
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=df['timestamp'], 
        y=df['speed'], 
        mode='lines', 
        name='Speed',
        line=dict(color='blue')
    ))
    
    shapes = []
    for idx, row in df.iterrows():
        # print (idx)
        if row['alert']:  # Check if alert is True
            shape = dict(
                type="line",
                x0=row['timestamp'], y0=0,
                x1=row['timestamp'], y1=df['speed'].max(),
                line=dict(color=color_map[row['band']], width=2, dash='dot')
                )  # Adjust color and width as needed
            
            shapes.append(shape)
            
    # Customizing the plot
    fig.update_layout(
        title='Speed over Time with Alerts',
        xaxis_title='Time',
        yaxis_title='Speed (units)',
        showlegend=True,
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightPink',
            rangeslider=dict(
                visible=True
            )
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=0.1, 
            gridcolor='LightPink'
        ),
        shapes=shapes  # Add shapes for alerts
    )

    # Update x-axis and y-axis ranges based on relayoutData
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        fig.update_xaxes(range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']])
    if relayoutData and 'yaxis.range[0]' in relayoutData:
        fig.update_yaxes(range=[relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']])
        
    return fig

@app.callback(
    Output('image-display', 'src'),
    Input('speed-plot', 'clickData')
)
def display_image(clickData):
    if clickData is not None:
        point = clickData['points'][0]
        clicked_time = pd.to_datetime(point['x'])
        
        alert_times = df[df['alert']]['timestamp']
        alert_times_np = alert_times.values.astype('datetime64[ns]')
        clicked_time_np = np.datetime64(clicked_time)
        closest_time_index = np.argmin(np.abs(alert_times_np - clicked_time_np))
        closest_time = alert_times.iloc[closest_time_index]
        
        row = df[df['timestamp'] == closest_time].iloc[0]
        imgpath = row['path']
        image_path = f"/home/achintya-trn0175/Desktop/alertsystem/22June/imgs/{imgpath}.jpg"
        
        print(f"image_path: {image_path}")  # Check the constructed image path
        
        if os.path.exists(image_path):
            encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
            print("Image encoded successfully")  # Check if the image was encoded correctly
            return 'data:image/jpeg;base64,{}'.format(encoded_image)
        else:
            print("Image not found at path:", image_path)  # Check if image path exists
        
    print("No click data available or image not processed correctly")
    return None

def log_alert_result(timestamp, correct, incorrect, image_path):
    log_entry = {
        'timestamp': timestamp,
        'correct_alert': correct,
        'incorrect_alert': incorrect,
        'image_path': image_path
    }
    log_file = 'alert_log.csv'
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

@app.callback(
    [Output('dummy-output', 'children'),
     Output('notification-dialog', 'displayed'),
     Output('notification-dialog', 'message')],
    [Input('correct-button', 'n_clicks'),
     Input('incorrect-button', 'n_clicks')],
    [State('speed-plot', 'clickData')]
)
def handle_alert_buttons(correct_clicks, incorrect_clicks, clickData):
    if clickData is not None:
        point = clickData['points'][0]
        clicked_time = pd.to_datetime(point['x'])
        
        alert_times = df[df['alert']]['timestamp']
        alert_times_np = alert_times.values.astype('datetime64[ns]')
        clicked_time_np = np.datetime64(clicked_time)
        closest_time_index = np.argmin(np.abs(alert_times_np - clicked_time_np))
        closest_time = alert_times.iloc[closest_time_index]
        
        row = df[df['timestamp'] == closest_time].iloc[0]
        image_path = row['path']
        
        if image_path != 'none':
            ctx = callback_context
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            correct = 1 if button_id == 'correct-button' else 0
            incorrect = 1 if button_id == 'incorrect-button' else 0
            log_alert_result(closest_time, correct, incorrect, image_path)
            notification_message = 'Correct Alert registered!' if correct else 'Incorrect Alert registered!'
            return '', True, notification_message
    return '', False, ''

if __name__ == '__main__':
    app.run_server(debug=True,port=8057)
