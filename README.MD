Welcome to the Swiss Vacation Planner Web Application!

Its purpose is to help individuals or groups of up to 10 people plan a personalised Swiss vacation itinerary that fits everyone's preferences and schedule.

After downloading the project folder in the same structure as provided, the connection to the preloaded activity database works effortlessly. Apart from a few personal preferences, no further data must be provided by the user(s). Open a terminal or PowerShell window. Before launching the app, navigate to the folder Swiss_Vacation_Planner via the cd command. Then first install the required packages via pip install -r requirements.txt. You can then launch the application by running: streamlit run app.py. Your browser will automatically open the application.

How it works:

The application is divided into four clear steps. 
First, the number of travellers in your group  is set. Then, each user selects their activity preferences — such as preferred categories, fitness level, and trip duration. A personalised set of recommended activities is generated from the database based on the group's combined preferences using a KNN-based recommendation model. Finally, after all preferences have been submitted, click "Generate My Itinerary" and a full day-by-day Swiss vacation plan will be created. It will include real-time weather forecasts for your destination, a timetable, and an interactive map showing all recommended activities.

You've done it — enjoy your Swiss adventure!