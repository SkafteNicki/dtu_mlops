import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid, train_test_split

np.random.seed(123)
OPTUNA = False

data = datasets.load_digits()
X,y = data['data'], data['target']
N = X.shape[0]
print(f"Datamatrix size {X.shape}. Label size {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

if not OPTUNA:
    kf = KFold(n_splits=5)
    
    N_ESTIMATORS = [1, 5, 10, 20, 50, 100, 200]
    MAX_DEPTH = [1, 5, 10, 20, 100]
    
    kf = KFold(n_splits=5)
    
    # we are going to do a full grid search
    params = ParameterGrid({'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH})
    
    scores = []
    for p in params:
      c = RandomForestClassifier(**p)
      scores.append([])
      for train_index, val_index in kf.split(X_train):
        x_t, x_v = X_train[train_index], X_train[val_index]
        y_t, y_v = y_train[train_index], y_train[val_index]
    
        c.fit(x_t, y_t)
        
        preds = c.predict(x_v)
    
        acc = accuracy_score(y_v, preds)
    
        scores[-1].append(acc)
    
    scores_mean = [np.mean(s) for s in scores]
    scores_std = [np.std(s) for s in scores]
    
    idx = np.argmax(scores_mean)
    print(f"Best parameter combination: {params[idx]}")
    classifier = RandomForestClassifier(**params[idx])
    classifier.fit(X_train, y_train)
    
    preds = classifier.predict(X_test)
    final_acc = accuracy_score(y_test, preds)
    print(f'Final score to report: {final_acc}')

##################### Here starts the exercise #####################
else:
    import optuna
    
    def objective(trial):
        # fill in this. Given a trial it should
        # 1. suggest a set of hyperparameters (HINT: use trial.suggest_discrete_uniform )
        # 2. train a random forest using the hyperparameters
        # 3. evaluate the trained random forest
        return val_acc
    
    # call the optimizer
    study = optuna.create_study()
    study.optimize(objective, n_trials=100) 