# Data 
The datasets as collected for each step can be found in the respective folder for each source channel. Sets are further split between annotated and raw data. 

### Dataset NEWS
News data can be imported by doing import {bloomberg, newsbitcoin, reuters, cnbc, wsj, coindesk} at the start of your Python script. 

The news data set has the following attributes:
- Title [‘’]
- Author [‘author’][‘name’]
- Timestamp [‘timestamp’] (datetime object so make sure to import that) 
- Body [‘articleText’]
- Source [‘publisher’]
- URL [‘url’]

You’d access the title of the first article in the Bloomberg set by bloomberg.data[0][‘title’]. 0 Is the article index and will loop over all articles if you specify it.  

### Datasets REDDIT
Stored as CSV Files
- Author ['author']
- Post text ['summary']
- Source Reddit ['source']
- Post score ['score']
- Timestamp ['timestamp']

### Datasets IRC
Stored as CSV files. Following attributes:
- Message body [‘summary’]
- Timestamp [‘timestamp’] (datetime object)
-  Author [‘author’]

### Dataset FORUM
Stored as CSV 
- Timestamp (datetime) ['timestamp']
- Subforum ['source']
- Author ['author']
- Comment count ['comment_count']
- Author activity ['author_activity']
- Post text ['summary']

## Annotated datasets
The training datasets in each subfolder contain all ratings as provided by the Amazon Mechanical Turk service, the average of the ratings, and the majority vote. The training datasets further contains cleaned strings, POS-tagged texts, and texts with stop words removed. 

# Classifiers
### Running the classifiers 
Classifiers are split up per channel, as the datastructures vary. Lots of repeated code as a result for now. 
- python classifyNews.py
- python classifyReddit.py
- python classifyForum.py
- python classifyIRC.py


## Causality workfiles
The default configuration uses the best respective classifier (as per the paper) for each source channel. Running the classifier will also match classified sentiment to market data (with a specific lag) and calculate and output the correlations. The sentiment + market data match was manually exported to EViews9 for the Granger causality test. The CSV files that were used to calculate causality are included in the /Granger folder. 
