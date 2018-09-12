# twitter_sentiment_challenge

##Overview

This is the code for the Twitter Sentiment Analyzer challenge for 'Learn Python for Data Science #2' by @Sirajology on [YouTube](https://youtu.be/o_OZdbCzHUA). The code uses the [tweepy](http://www.tweepy.org/)  library to access the Twitter API and the [TextBlob](https://textblob.readthedocs.io/en/dev/) library to perform Sentiment Analysis on each Tweet. We'll be able to see how positive or negative each tweet is about whatever topic we choose. 

##Dependencies

* tweepy (http://www.tweepy.org/)
* textblob (https://textblob.readthedocs.io/en/dev/)

Install missing dependencies using [pip](https://pip.pypa.io/en/stable/installing/)

##Usage

Once you have your dependencies installed via pip, run the script in terminal via

```
python demo.py
```

##Challenge

Last week, there was the french Republicans Primary debate on television.
I tried to apply Siraj's sentiment analysis to this night of debate in order to compare the seven different candidates.
Here is the result of the analysis : 

Mean Sentiment Polarity in descending order :
* Poisson : 0.180
* Fillon : 0.113
* Juppe : 0.098
* Sarkozy : 0.057
* Cope : 0.036
* Le Maire : 0.007
* Kosciusko : 0.007

##Credits

This code is forked from Siraj
