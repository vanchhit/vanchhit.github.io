# Import necessary libraries for HTTP requests, data handling, datetime operations, and plotting
def thegraph():
    # Importing data from APIs
    import requests
    # Manage dataframes and analysis
    import pandas as pd
    import numpy as np
    # Date, time and time zones
    import datetime
    import pytz
    # Matplotlib for usual graph
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    from io import BytesIO
    from IPython.display import display
    # Interpolation for smooth curves
    from scipy.interpolate import CubicSpline
    # Plotly for web published graphs
    import plotly.express as px
    import plotly.graph_objects as go

    # Define the API endpoint for retrieving energy data

    # API endpoint for the /signal route
    url = "https://api.energy-charts.info/signal?country=de"

    plt.figure(figsize = (10, 5))
    # Make a GET request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the entire JSON response
        data = response.json()

        # Extract relevant information if available
        unix_seconds = data.get("unix_seconds")
        share = data.get("share")
        signal = data.get("signal")
        substitute = data.get("substitute")
    else:
        print(f"Error: {response.status_code}")


    # Relevant time zones
    berlin = pytz.timezone("Europe/Berlin")
    date_time = [datetime.datetime.fromtimestamp(i, tz = berlin) for i in unix_seconds]

    today_date = datetime.datetime.now().date()

    if len(date_time) > 96:
        predict_date = date_time[96:]
        date_time = date_time[:96]
        full_data = [predict_date, share[96:], signal[96:]]
    else:
        full_data = [date_time, share, signal]

    signal_data = pd.DataFrame(list(zip(*full_data)), columns = ["Date and Time", "Renewable Share", "Signal"])

    # Assuming signal_data and 'Date and Time' are already defined
    xes = np.array([i.hour + i.minute / 60 for i in signal_data['Date and Time']])
    y = signal_data["Renewable Share"]

    # Create a cubic spline model
    cs = CubicSpline(xes, y)

    # Generate new x values at 1-minute intervals
    x_new = np.arange(xes.min(), 24, 1/60)

    # Interpolate y values for these new x values
    predicted_share = cs(x_new)

    # Mapping of traffic signal to color
    colors = {
        -1: 'brown',
        0: 'red',
        1: '#ebeb00',
        2: 'green'
    }

    # Map each signal to its corresponding color
    signal_data["Traffic Light"] = [colors[i] for i in signal_data["Signal"]]

    #Finding the "greenest" time to use electricity
    start_time = signal_data['Date and Time'].min()
    max_avg = 0
    max_avg_start_times = []
    running_avg = []

    # Calculate average for each 2-hour interval
    for i in range(len(signal_data["Signal"])):
        # Create a mask for the current 2-hour interval
        mask = (signal_data['Date and Time'] >= start_time) & (signal_data['Date and Time'] <= start_time + pd.Timedelta(hours=2))
        mask_2 = (signal_data['Date and Time'] >= start_time) & (signal_data['Date and Time'] <= start_time + pd.Timedelta(hours=0.5))

        # Calculate the average for the current interval
        avg_value = signal_data.loc[mask, 'Signal'].mean()
        gradient_increase = signal_data.loc[mask_2, 'Signal'].mean()
        running_avg.append(gradient_increase)

        if avg_value > max_avg:
            max_avg = avg_value
            max_avg_start_times = [start_time]
        elif avg_value == max_avg:
            max_avg_start_times.append(start_time)

        # Update the start time for the next iteration by adding 15 minutes
        start_time += pd.Timedelta(minutes=15)

    best_times = {
        "2 hour avg": max_avg,
        "Start Time": max_avg_start_times
    }

    #Merging best times

    start_time = signal_data['Date and Time'].min()
    max_avg = -1
    max_avg_start_times = []
    max_avg_intervals = [(start_time, start_time + pd.Timedelta(hours=2))]


    # Calculate average for each 2-hour interval
    for i in range(len(signal_data["Signal"]) - (int(2/0.25)-1)):
        # Create a mask for the current 2-hour interval
        mask = (signal_data['Date and Time'] >= start_time) & (signal_data['Date and Time'] <= start_time + pd.Timedelta(hours=2))

        # Calculate the average for the current interval
        avg_value = signal_data.loc[mask, 'Signal'].mean()

        if avg_value > max_avg:
            max_avg = avg_value
            max_avg_intervals = [(start_time, start_time + pd.Timedelta(hours=2))]
        elif avg_value == max_avg:
            # Check if the current interval overlaps with the last interval in max_avg_intervals
            last_interval_start, last_interval_end = max_avg_intervals[len(max_avg_intervals)-1]
            if start_time < last_interval_end:
                # Merge overlapping intervals
                max_avg_intervals[-1] = (last_interval_start, max(start_time + pd.Timedelta(hours=2), last_interval_end))
            else:
                max_avg_intervals.append((start_time, start_time + pd.Timedelta(hours=2)))

        # Update the start time for the next iteration by adding 15 minutes
        start_time += pd.Timedelta(minutes=15)

    #Finding the worst times

    start_time = signal_data['Date and Time'].min()
    min_avg = 2
    min_avg_start_times = []
    min_avg_intervals = [(start_time, start_time + pd.Timedelta(hours=2))]


    # Calculate average for each 2-hour interval
    for i in range(len(signal_data["Signal"]) - (int(2/0.25)-1)):
        # Create a mask for the current 2-hour interval
        mask = (signal_data['Date and Time'] >= start_time) & (signal_data['Date and Time'] <= start_time + pd.Timedelta(hours=2))

        # Calculate the average for the current interval
        avg_value = signal_data.loc[mask, 'Signal'].mean()

        if avg_value < max_avg:
            max_avg = avg_value
            max_avg_intervals = [(start_time, start_time + pd.Timedelta(hours=2))]
        elif avg_value == max_avg:
            # Check if the current interval overlaps with the last interval in max_avg_intervals
            last_interval_start, last_interval_end = max_avg_intervals[len(max_avg_intervals)-1]
            if start_time < last_interval_end:
                # Merge overlapping intervals
                max_avg_intervals[-1] = (last_interval_start, max(start_time + pd.Timedelta(hours=2), last_interval_end))
            else:
                max_avg_intervals.append((start_time, start_time + pd.Timedelta(hours=2)))

        # Update the start time for the next iteration by adding 15 minutes
        start_time += pd.Timedelta(minutes=15)

    #potential savings

    peak = signal_data["Renewable Share"].max()
    trough = signal_data["Renewable Share"].min()

    peak_time = signal_data['Date and Time'][signal_data['Renewable Share'] == peak].values
    trough_time = signal_data['Date and Time'][signal_data['Renewable Share'] == trough].values

    potential_savings = ((peak - trough)/peak)*100
    potential_savings_str = str(int(round(potential_savings, 0))) +"%"

    peak_s = str(int(round(peak, 0))) +"%"
    trough_s = str(int(round(trough, 0))) +"%"

    # Create a cubic spline model
    cs2 = CubicSpline(xes, running_avg)

    # Interpolate y values for these new x values
    predicted_signal = cs2(x_new)

    # URL of the image
    image_url = "https://www.httpsimage.com/v2/f2ad943c-cb2b-4f77-8e44-3ef6dbc8e513.png"

    # Make a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the image using PIL
        img = Image.open(BytesIO(response.content))

    else:
        print(f"Error: {response.status_code}")

    highest = predicted_share.argmax()
    lowest = predicted_share.argmin()
    print(highest, lowest)

    ppeak = max(predicted_share)
    ptrough = min(predicted_share)

    #The final Graph + Whatsapp Message text
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.image as mpimg

    dt = signal_data["Date and Time"][0]
    engde = {
        "Monday" : "Montag",
        "Tuesday" : "Dienstag",
        "Wednesday" : "Mittwoch",
        "Thursday" : "Donnerstag",
        "Friday" : "Freitag",
        "Saturday" : "Samstag",
        "Sunday" : "Sonntag"
    }
    print(engde[dt.strftime('%A')])

    # Define color mapping
    colors = [(1, 0, 0), (0.9, 0.9, 0.1), (0, 0.55, 0.1)]  # Red, Yellow, Green
    n_bins = [0, 1, 2]  # Define bin edges for the gradient

    # Create colormap
    cmap_name = 'custom_gradient'
    gradient = np.linspace(0, 1, 256)
    n_bins = np.array(n_bins)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Plot the bar chart with the specified colors
    plt.bar(x_new, predicted_share, color=cmap([i/2 for i in predicted_signal]), width = 0.1) # Divided by 2 because colormap maps from 0 to 1, we have to find a way of normalising negative values in the near future.

    ax = plt.gca()

    a,b = ax.get_ylim()

    # Convert 'predicted_signal' to colors
    colors = np.array(['brown', 'red', '#ebeb00', 'green'])  # Map your signal values to colors
    signal_colors = np.digitize(predicted_signal, bins=[-0.5, 0.5, 1.5, 2.5], right=True)
    color_mapped = colors[signal_colors]

    # Create the base figure
    fig = go.Figure()

    # Add the bar chart for renewable share with dynamic coloring based on 'predicted_signal'
    for i, x in enumerate(x_new):
        fig.add_trace(go.Bar(x=[x], y=[predicted_share[i]], marker_color=color_mapped[i], showlegend=False))

    # Add horizontal lines for peak and trough
    fig.add_trace(go.Scatter(x=[0, x_new[highest]], y=[ppeak, ppeak], mode='lines', line=dict(color='black', dash='dash'), name='Peak'))
    fig.add_trace(go.Scatter(x=[0, x_new[lowest]], y=[ptrough, ptrough], mode='lines', line=dict(color='black', dash='dash'), name='Trough'))

    # Add vertical lines for peak and trough
    fig.add_vline(x=x_new[highest], line=dict(color='black', dash='dash'), line_width=1)
    fig.add_vline(x=x_new[lowest], line=dict(color='black', dash='dash'), line_width=1)

    # Customizing the layout
    fig.update_layout(
        title=f"% Erneuerbare im deutschen Strommix, {dt.strftime('%d.%m.%y')}",
        xaxis=dict(title='Uhrzeit', tickmode='array', tickvals=[0, 4, 8, 12, 16, 20, 24], ticktext=['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']),
        yaxis=dict(title='% Erneuerbare Energie'),
        template='plotly_white'
    )

    # Add annotations for the peak and trough
    fig.add_annotation(x=x_new[highest], y=ppeak, text=f"Peak: {ppeak}%", showarrow=True, arrowhead=1, ax=20, ay=-30)
    fig.add_annotation(x=x_new[lowest], y=ptrough, text=f"Trough: {ptrough}%", showarrow=True, arrowhead=1, ax=20, ay=30)

    plt.axhline(y = ppeak, color='black', linestyle='--', label=peak, xmax = highest/len(x_new), alpha = 0.5)
    plt.axhline(y = ptrough, color='black', linestyle='--', label=trough, xmax = lowest/len(x_new), alpha = 0.5)
    plt.axvline(x = x_new[highest], ymax = ppeak/b, color='black', linestyle='--', alpha = 0.5)
    plt.axvline(x = x_new[lowest], ymax = ptrough/b, color='black', linestyle='--', alpha = 0.5)

    plt.xlim(0, 24)

    # Set the position to the bottom right corner
    imagebox = OffsetImage(img, zoom=0.04)  # Adjust the zoom factor as needed
    ab = AnnotationBbox(imagebox, xy=(0.95, 0.0657), xycoords='axes fraction', frameon=False)

    # Add color bar for reference

    # Set labels and title
    custom_ticks = [0, 4, 8, 12, 16, 20, 24]
    custom_labels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    plt.xticks(custom_ticks, custom_labels)
    plt.xlabel('Uhrzeit')
    plt.ylabel('% Erneuerbare Energie')
    plt.title(f"% Erneuerbare im deutschen Strommix, {dt.strftime('%d.%m.%y')}")

    # Show the plot
    plt.subplot().add_artist(ab)

    #IMPORTANT: this code adds one day to the date so if we run the program on the day of the forcast it will show the wrong date
    #Program must be run the day before!
    predicted_date = datetime.datetime.now()+ datetime.timedelta(days = 1)
    s_predicted_date = predicted_date.strftime('%Y-%m-%d')

    print(f"\n*{engde[dt.strftime('%A')]}, {dt.strftime('%d.%m.')}*"
        "\nEinsparpotential: ", potential_savings_str,""
        "\n✅Beste Zeit:", "Uhr, " + peak_s,
        "\n❌Vermeiden: ", "Uhr, ", trough_s,
        "\nUm zu pausieren, antworte Stopp")

    plt.savefig("graph_img.svg")

thegraph()