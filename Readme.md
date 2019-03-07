# Assignment Title : Yelp Review Classifier for our new ranking scheme

The main goal of this assignment is to perform sentiment classification of Yelp Reviews. In this assignment, we use Yelp review dataset of user reviews. Yelp is an application to provide the platform for customers to write reviews and provide a star-rating. A research indicates that a one-star increase led to 59% increase in revenue of independent restaurants. 

To build a model to predict if review is positive, negative or neutral, following steps are performed.

1)Importing Dataset
2)Preprocessing Dataset
3)Pipelining
4)Training and Classification
5)Findings, Analysis and Conclusion

## Getting Started

We place the "yelp_academic_dataset_review.csv", "positive-words.txt" and "negative-words.txt" in your working directory to easily import the files into your code. 

### The Dataset

Our data contains 45,000 reviews, with the following information for each one

1)business_id (ID of the business being reviewed)
2)date (Day the review was posted)
3)review_id (ID for the posted review)
4)stars (1–5 rating for the business)
5)text (Review text)
6)type (Type of text)
7)user_id (User’s id)
8){cool / useful / funny} (Comments on the review, given by other users)

#### Importing the Dataset

```
import pandas as pd

data =  pd.read_csv(r'yelp_academic_dataset_review.csv',encoding = "utf8",keep_default_na=False,nrows=200000)
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)
data.to_sql('yelp_reviews',con=engine)
```

Firstly, let’s import the necessary Python libraries. NLTK is pretty much the standard library in Python library 
for text processing, which has many useful features. Today, we will just use NLTK for stopword removal.

Let’s see how we can go about analysing this dataset using Pandas, Pipelining, and Scikit-learn.

### Text Pre-Processing

Each review undergoes through a preprocessing step, where all the vague information is removed.

	We break the sentences into word tokens
	We remove the stop words (such as "the", "a", "an", "in" etc), special characters from the tokens

 Below is an example review from Yelp Review DataSet which has undergone pre-processing.

```
['Ate', 'Saturday', 'morning', 'breakfast', 'Pine', 'Cone', 'Friendlyquick', 'service', 'normal', 'prices', 'new', 'year', 'special', 'allyoucaneat', 'pancakes', '212', 'added', 'breakfast', 'sausage', 'patty', '300', 'Father', 'typical', 'two', 'eggs', 'toast', 'hashbowns', 'seem', 'fresh', 'cut', 'sausage', 'links', 'Food', 'tasty', 'major', 'qualms', 'hashbrowns', 'need', 'bit', 'cooking', 'coffee', 'needs', 'bit', 'bite', 'nice', 'assortment', 'bakes', 'goods', 'massive', 'large', 'head', 'creampuffs', 'eclairs', 'took', 'blueberry', 'muffin', 'go', 'enjoyed', 'Quick', 'exit', 'I94just', 'northeast', 'Madison']
```

After the preprocessing, the tokens are then compared with a set of 'positive' and 'negative' words to assign scores. If token is present in Positive words, we assign a score of '1', for negative a score of '-1' is assigned and if token is not present in both sets, then a score of '0' is assigned, which is considered as neutral.
Based on these token scores, the mean score for each review is calculated. The value of mean score helps us to label each review as "Positive", "Negative" or "Neutral"

## Building the Model

Pipelining is used to build our Logistic Regression based model. The modules of the pipeline are:
1) Countvectorizer
2) Tfidftransformer
3) Logistic Regression Algorithm

```
Pipeline(memory=None,
     steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words='english',
        ...ty='l2', random_state=None, solver='newton-cg',
          tol=0.0001, verbose=0, warm_start=False))])
```

### Training and Testing the Data

 It’s time to split our X and y into a training and a test set using train_test_split from Scikit-learn. 
We will use 25% of the dataset for testing. The training data was then fed to the Pipeline to train the model.

```
x_train,x_test,y_train,y_test = train_test_split(dataset,labels,test_size=0.25,random_state=108)
```

### Training our model

Pipelining is a specialised version designed more for text documents. 
Let’s build a Pipelining model and fit it to our training set (X_train and y_train). 


### Results

From the above algorithm modelling, we can see that:

```
|   Predicted |   Actual | Business_id            | Result   |
|-------------+----------+------------------------+----------|
|     2.74692 |  3.27635 | jf67Z1pnwElRSXllpQHiJg | correct  |
|     3.43436 |  3.47659 | FV16IeXJp2W6pnghTz2FAw | correct  |
|     2.82236 |  3.24557 | hW0Ne_HTHEAgGF1rAdmR-g | correct  |
|     3.37163 |  4.31154 | JokKtdXU7zXHcr20Lrk29A | wrong    |
|     3.19733 |  4.06886 | 4UVhuOLaMm2-34SrW8y-ag | wrong    |
|     3.31265 |  4.48724 | SDwYQ6eSu1htn8vHWv128g | wrong    |
|     3.19167 |  4.18182 | K8pM6qQdYu5h6buRE1-_sw | wrong    |
|     2.68302 |  2.51402 | KJnVuzpveyDrHARVNZaYVg | correct  |
|     3.04538 |  4.33168 | c1yGkETheht_1vjda7G5sA | wrong    |
|     3.27787 |  4.27959 | L9UYbtAUOcfTgZFimehlXw | wrong    |
```

## Findings

From the below predictions, we can see that predictions are biased towards positive reviews. We can see that the dataset has more positive reviews as compared to negative reviews. 
I think we can fix it by normalizing the dataset to have equal number of reviews - thereby removing the bias.

```
              precision    recall  f1-score   support

          -1       0.69      0.71      0.70      3761
           0       0.56      0.55      0.56      4022
           1       0.72      0.71      0.72      3456

   micro avg       0.66      0.66      0.66     11239
   macro avg       0.66      0.66      0.66     11239
weighted avg       0.65      0.66      0.65     11239
```



