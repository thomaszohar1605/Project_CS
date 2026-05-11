Welcome to the Swiss Vacation Planner Web Application!

Its purpose is to help individuals or groups of up to 10 people plan a personalised Swiss vacation itinerary that fits everyone's preferences and schedule.

After downloading the project folder in the same structure as provided, the connection to the preloaded activity database works effortlessly. Apart from a few personal preferences, no further data must be provided by the user(s). Open a terminal or PowerShell window. Before launching the app, navigate to the folder Swiss_Vacation_Planner via the cd command. Then first install the required packages via pip install -r requirements.txt. You can then launch the application by running: streamlit run app.py. Your browser will automatically open the application.

How it works:

The application is divided into four clear steps. 
First, the number of travellers in your group  is set. Then, each user selects their activity preferences — such as preferred categories, fitness level, and trip duration. A personalised set of recommended activities is generated from the database based on the group's combined preferences using a KNN-based recommendation model. Finally, after all preferences have been submitted, click "Generate My Itinerary" and a full day-by-day Swiss vacation plan will be created. It will include real-time weather forecasts for your destination, a timetable, and an interactive map showing all recommended activities.

You've done it — enjoy your Swiss adventure!

1.	A problem that is solved by the application is clearly stated (e.g., business or consumer use case)
0 - Problem, purpose remain unclear
1 - Problem rather too simplistic such as "show stock price of XXX  over the last year"
2 - Interesting and relevant problem such as "what can I cook if I have the following things in my fridge"
3 - Interesting and relevant problem that is also innovative, not yet solved (or at least not in this interesting way), and potentially very useful

2.	The application uses some data that is loaded via an API and/or provided via a database
0 - does not exist
1 - some API is used just for the sake of using an API (such as currency conversion)
2 - the API use is a core aspect of the use case
3 - in addition, several APIs are used in a sophisticated way to enable the core ideas of the app

3.	The application visualizes some data that serves the use case
0 - does not exist (a table with values is not a data visualization)
1 - some data is visualized just for the sake of visualizing data (e.g. a simple line diagram of stock prices)
2 - the specific data visualization is a core aspect of the use case
3 - in addition, vizualization goes beyond standard line/bar charts

4.	The application allows for some user interaction
0 - does not exist (app is static or buttons/switches have no effects)
1 - some simple button/switch but it is not core to the use case (e.g. changing color)
2 - user can purposefully interact with the app to solve the problem
3 - advanced multi-page app or advanced interactions such as filtering/manipulating data in several ways relevant to the problem
5.	The application implements some machine learning
0 - does not exist (using the openAI API does not implement ML)
1 - some simple model is trained and used for the sake of having ML but it hardly helps solve the problem
2 - some simple model is trained and used; it supports solving the problem
3 - rather advanced (in the FCS context) model/approach with interesting dataset at the core of the problem to be solved

6.	The source code is well documented by comments in the source code
0 - does not exist or is not useful at all
1 - some comments here and there but not systematic and not sufficient to understand the underlying ideas of the code
2 - all blocks in the code are systematically commented; helps to quickly understand the logic
3 - in addition, all files/functions have well-formed doc-strings

7.	The contributions of each team member are documented (e.g., contribution matrix)
0 - does not exist ("every member was involved in everything" is not sufficient)
1 - simple but rather high level contribution matrix (e.g. coding, video, documentation) 
2 - detailed contribution matrix providing an idea of HOW the project has been structured and WHO was HOW MUCH involved in WHAT
3 - in addition, it becomes clear WHY the project has been organized in that way, what has been the underlying idea/logic; levels of engagement (e.g., "lead", "support") are explained
8.	The result is presented and demoed in a 4-minute video
0 - does not exist or demo is missing
1 - the underlying ideas of app and implementation remain vague, the video may also be too short or longer than 4 minutes
2 - nice video that presents idea, implementation, use of the app and the other aforementioned aspects in a compact and understandable way, must not be longer than 4 min.
3 - in addition, the video is really high quality or very original or very much fun to watch or ... something that you really think outside people should not miss
