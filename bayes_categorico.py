from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from math import inf
import pandas as pd
from copy import deepcopy


def train_bayes_categorico(n_bins, X_train, y_train, X_val, y_val, X_test, y_test):
  best_val_error = inf

  errors = []

  for bins in n_bins:
    kbdisc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    kbdisc.fit(X_train)

    X_train_discreto = kbdisc.transform(X_train.copy())
    X_val_discreto = kbdisc.transform(X_val.copy())
    X_test_discreto = kbdisc.transform(X_test.copy())

    clf = CategoricalNB(min_categories=bins)
    clf.fit(X_train_discreto, y_train)

    predict_train = clf.predict(X_train_discreto)
    predict_val = clf.predict(X_val_discreto)
    predict_test= clf.predict(X_test_discreto)

    actual_train_error = 1 - accuracy_score(y_train, predict_train)
    actual_val_error = 1 - accuracy_score(y_val, predict_val)
    actual_test_error = 1 - accuracy_score(y_test, predict_test)

    errors.append([actual_train_error, bins, "Bayes - Train"])
    errors.append([actual_val_error, bins, "Bayes - Validation"])
    errors.append([actual_test_error, bins, "Bayes - Test"])

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_bins = bins
      best_clf = deepcopy(clf)
      best_kbdisc = deepcopy(kbdisc)

  df_errores = pd.DataFrame(errors, columns = ['Error', 'Bins', 'Clase'])

  return best_bins, best_clf, best_kbdisc, df_errores