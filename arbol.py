from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree


class Arbol:
  def entrenar(self, df):
    X, y = df[[0,1]], df['Clase']
    clf = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=0.005,random_state=0,min_samples_leaf=5)
    clf.fit(X, y)

    return clf

  def predecir(self, df_test, clf):
    X = df_test[[0,1]]
    prediccion = clf.predict(X)
    df = df_test.copy(deep = True)
    df['Clase'] = prediccion

    return df


  def test(self):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=0.005,random_state=0,min_samples_leaf=5)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)