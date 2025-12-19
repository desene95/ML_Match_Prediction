# ML_Match_Prediction

AvgH = Average odds for Home win
AvgD = Average odds for Draw
AvgA = Average odds for Away win

Why they matter:

Odds reflect collective bookmaker + market intelligence

Lower odds = higher implied probability

They are very powerful predictors

train_test_split splits your dataset into two parts:

Training set → the model learns from this

Test set → the model is evaluated on this (unseen data)

X = match data (odds, stats, etc.)

y = result label (home win = 1, else = 0)

What does this line do
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

“Split my data so that:

the first 80% of matches are used for training

the last 20% of matches are used for testing

keep the original order of matches”

test_size=0.2

    20% of your data → test set

    80% → training set

After first training:
There were 32 games
12 Home wins
20 Non-Home wins

Model predicted 20 Home wins
    10 were actual home wins, other 10 were false

Model over predicts home wins

Recall = “Of all the matches that actually were home wins, how many did the model identify correctly?”
Precision = “Of all matches the model predicted as home wins, how many were actually home wins?”


print(len(X_train), len(X_test))
128 32
Model training/learnign from 128 games in the data set
Evaluating 32 games

test_size=0.2 - use 20% of the data set for evaluation. The other 80% will be used to train. Increasing this value means the model has limited data to train/learn from

# Py Files
ML_Train_1.py - Only uses 160 games (All EPL games so far this season). No rolling average
ML_Train_2.py - Over 3k games; Rolling averages implemented; biased toward class 0 (it predicts class 0 more confidently than class 1).

Next steps for improvement

Add more informative features:

Away team form

Head-to-head history

Odds from betting markets

Player injuries / lineups