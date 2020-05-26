# TweetDepressionDetector

Detecting mental illness in the field of social media can be a very difficult task sincethe  definition  of  these  mental  illness  canvery from person to person. Over 300 million people of all ages suffer from depression worldwide. WHO classifies  depression  broadly  into  3  categories:  mild,  moderate  and  severe.   De-pression can be caused due to Stress, anxiety,  past  traumas  and  can  cause  seriousphysical  illness  which  is  detrimental  inthe  long  run.   In  this  project  we  trainedtwo types of classifiers to classify tweetsas depressive or not.   We used data from twitter pages related to depression and la-beled them as 1(depressive tweet) and ex-tract  tweets  from  Sentiment  140  data  setfor the non depressive tweets.  Using thisdata set, we trained our algorithms to predict if a given user has depression or not.We achieved an accuracy of  99% test setaccuracy for classifier 1 which is used topredict  if  there  is  depression  content  in the tweet, and 80% test set accuracy for classifier 2 which is used to classify if the given user is at risk of depression

Group Size:3

Step 1 --- Manually annotated the depressive_tweets dataset, It has  1805  0's and 1879 1's.

Step 2 -TfIdf vectorization

Step 2 -- train LinearSVM on datasetA
	  train Multinomial NB on dataset A
	  train LogisticRegression on dataset A 
	 
