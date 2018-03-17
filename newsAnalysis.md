
# Analysis: 

* BBC trends show the most positive polarity score
* CNN's tweets are the most neutral out of the lister News Channels, because its average compount score is closest to zero.
* The news channels do not show much change over the course of 100 tweets ago compared to current day.


```python
import pandas as pd
import tweepy
import numpy as np
import json
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
twitter_consumer_key = "ThPe6egpT3OJCiRfx7w7lGyVH"
twitter_consumer_secret = "EnGwBptSnjSKEjgIbOi3D4qtRvfjcgnCWm3pDGQscjs07qXbG2"
twitter_access_token = "969396072215818240-ZVyRxct50k8giGAXzKoAVd4uThbuxSu"
twitter_access_token_secret = "tgyVOelmg2LN0FdlbrW37rh55DvMtWzwxj7oQtYRkBy81"

auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Twitter accounts to read
news_user = ("@BBC","@CBSNews", "@CNN", "@FoxNews", "@NYTimes")
```


```python
# setting default values
users = []
sentiment = []
neg = []
pos = []
neut = []
users = []
num = []
dates = []
text = []
oldest_tweet = None
avg_sentiments = []

counter = 0
```


```python
for user in news_user:
   # 100 most recent tweets
    public_tweets = api.user_timeline(user, count=100, result_type="recent", max_id=oldest_tweet)
    
    counter = 1

    for tweet in public_tweets:
        # sentiment analysis on each tweet
        results = analyzer.polarity_scores(tweet["text"])

        # adding to arrays
        sentiment.append(results["compound"])
        neg.append(results["neg"])
        pos.append(results["pos"])
        neut.append(results["neu"])
        users.append(user)
        dates.append(tweet["created_at"])
        text.append(tweet["text"])
        num.append(counter)
        counter +=1
       
avgSentiment = np.mean(sentiment)
avg_sentiments.append(avgSentiment)
        
sentiment_df = pd.DataFrame({"Channel":users, 
                        "Date":dates,
                        "Compound":sentiment,
                        "Negative":neg, 
                        "Positive":pos,
                        "Neutral":neut, 
                        "Text":text,
                        "Tweet Ago": num})
```


```python
sentiment_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweet Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.0000</td>
      <td>Sat Mar 17 15:03:12 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>This year, over 250 landmarks across the globe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBC</td>
      <td>0.0000</td>
      <td>Sat Mar 17 14:43:43 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCRadio3: "Even if we found a complete th...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBC</td>
      <td>0.8625</td>
      <td>Sat Mar 17 14:00:30 +0000 2018</td>
      <td>0.000</td>
      <td>0.699</td>
      <td>0.301</td>
      <td>⛷❤️ George has autism and other health conditi...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBC</td>
      <td>0.3628</td>
      <td>Sat Mar 17 13:46:15 +0000 2018</td>
      <td>0.073</td>
      <td>0.784</td>
      <td>0.143</td>
      <td>RT @5liveSport: 'My mates didn't know I was pl...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBC</td>
      <td>0.4019</td>
      <td>Sat Mar 17 13:00:08 +0000 2018</td>
      <td>0.000</td>
      <td>0.881</td>
      <td>0.119</td>
      <td>Yes, Gary Oldman and @BBCEastEnders' Big Mo ar...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiment_df.to_csv("NewsMood.csv")
```


```python
len(text)
```




    500




```python
users = sentiment_df["Channel"].unique()
colors = ["yellow", "lightskyblue", "darkblue", "red", "green"]

for i in range(len(users)):
    plt.scatter(x=sentiment_df[sentiment_df["Channel"]==users[i]]["Tweet Ago"].values,
                y=sentiment_df[sentiment_df["Channel"]==users[i]]["Compound"].values,
                s = 90,#*sentiment_df[sentiment_df['User']==users[i]]['Tweets_Ago'].values,
                c = colors[i], label = users[i],
                alpha = .7, edgecolor = 'black', linewidth = .8)

plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.title("Sentiment Analysis of Media Tweets")
plt.legend(title="Media Sources", loc="upper right")
plt.grid()
plt.gcf().set_size_inches(15, 6)
plt.rcParams["axes.facecolor"] = "gainsboro"
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.ylim(-1, 1)
plt.savefig("Sentiment_Analysis.png")
plt.show()
```


![png](output_9_0.png)



```python
avg_sentiment = {"Channel": target_users, "Average Compound": avg_sentiments}
avg_sentiment_df = pd.DataFrame(avg_sentiment)
avg_sentiment_df
```


```python
x_values = np.arange(len(avg_sentiment_df))
plt.figure(figsize=(10, 7))
barlist=plt.bar(x_values, avg_sentiment_df["Average Compound"], alpha=0.5, align='center', width=1)
plt.xticks(x_values, avg_sentiment_df["Channel"], rotation="horizontal")
for i in range(len(barlist)):
    barlist[i].set_color(colors[i])
plt.ylabel("Tweet Polarity")
plt.title("Overall Media Sentiment Based On Twitter")
plt.savefig("Overall_Media_Sentiment.png")
plt.show()
```
