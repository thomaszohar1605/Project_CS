Changez avec la description de votre projet



Welcome to our Song Generator Web application!

Its purpose is to help a group of people of up to 10 people to find a playlist that fits everyone’s music taste.

After downloading the Project folder in the same structure as provided, the connection to the preloaded database works effortlessly. Apart from the song ratings no further data must be provided by the user(s). Open a terminal or PowerShell window. Before launching the code navigate to the location you've saved the folder 'Project_Spotify-main' in via the function 'cd'. Then please first install the requirements.txt file (required packages) via the 'pip install -r requirements.txt'. Then you can launch the actual application by calling the python-file app.py via: 'streamlit run app.py'. Your browser then will automatically launch the application.

To open our Web application on your browser of choice, go to: https://projectspotify.streamlit.app/

It is clearly divided into the four steps where the user can input his or her preferences. First, the number of users has to be determined. Afterwards, the Music Genre, to which type the Playlist is supposed to belong to, can be chosen. Than, a randomized set of 5 songs out of the database coherent with the selected Music Genre will be generated and displayed for rating. After each user has rated every song between 1 and 5 stars, one can click on the “Generate Final Playlist” Button, and a playlist will be created. It will adhere to the length and music genre chosen, and take all ratings of each member into consideration.

You’ve done it! You can now enjoy your newly created playlist with the whole group.

You'll find our Contribution Matrix and video in the folder additional content! The file build_db.py was the function we used to create the database and aren't used in the actual code. The CSV files: genre_with_main_identity.csv, reduced_features.csv and tracks_small.csv were used to build the database and are in the data folder as well. Please also take a look at the citations.txt, as we based our project on a dataset that was created in a scientific project.
