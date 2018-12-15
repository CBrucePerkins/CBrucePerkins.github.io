---
title: "Sentiment Analysis of IMDb Reviews"
date: 2018-01-28
tags: [machine learning, data science, convolutional neural network, natural language processing]
header: 
  image: "/images/sentiment/sent.jpg"
---

# Sentiment Analysis Through Natural Language Processing

Sentiment analysis is a method for extracting the sentiment (positive or negative) from natural language. It is an extremely useful tool when it comes to monitoring public opinion, gauging the reception of a firm’s advertisement campaign or, in the case of this project, classifying the sentiment of movie reviews on IMDb. The focus is sentiment analysis of movie reviews, but the principles generalize to all online reviews across most domains. [Statistical trends](https://www.business2community.com/infographics/impact-online-reviews-customers-buying-decisions-infographic-01280945 ) from 2015 show that 90% of consumers read online reviews, and 88% of them trust online reviews as much as personal recommendations. Being able to efficiently and accurately classify the sentiment of online reviews is an invaluable tool for businesses that wish to get ahead of public opinion.

So let's get to it. For this project I used the data from an old Kaggle competition: ["Bag of Words Meets Bags of Popcorn."](https://www.kaggle.com/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn/home ), and to spare my machine the time and effort I'll only be using the training set consisting of 25,000 reviews.

### The Data

The dataset provided contains three attributes: “id”, “sentiment” and “review”. “Id” shows an anonymized user id within the IMDb website’s database. “Sentiment” is a binary variable classifying the reviews sentiment as positive or negative using the integer 1 or 0 respectively. Finally the actual raw text review is contained in the column “review”. The data contains 25,000 observations of varying review length, with most of them being in the 100-200 word range:

<img src="{{ site.url }}{{ stie.baseurl }}/images/sentiment/distribution.png" alt="">

