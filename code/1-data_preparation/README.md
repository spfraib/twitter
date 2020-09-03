## Data preparation


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