# twitter

This Repo contains Jupyter notebooks allowing to collect, clean, and classify twitter data. It is organized in the following way:

(1) Imports: 

- Extract subsamples from a historical dataset of Tweets based on specific criteria

(2) Locations:

- Geocode Twitter users' account location, allowing to map twitter data to official statistics

- Reverse geocode tweets with geocoordinates

(3) Users: 

- Extract list of users whose account location was properly geocoded

- Lookup located users' profile using the Twitter API, identifying users whose account are not protected 

(4) Timelines:

- Download unprotected users' timeline from the Twitter API

(5) Tweets:

- Extract subsample of Tweets located in a given region

(6) Mentions:

- Count mentions of specific keywords within each tweet

(7) Classification:

- Compute similarity of each tweet to a given sentence, allowing to find semantically similar tweets

- Feed a sample of tweets into a Qualtrics survey to create labels on Amazon Mechanical Turk

- Classify tweets based on the labor market status of Twitter users