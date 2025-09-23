import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    cmodel = CalibratedClassifierCV(model, method="sigmoid")
    cmodel.fit(X_train, y_train)

    y_proba = cmodel.predict_proba(X_test)
    print("predicted probabilities:\n",[int(np.argmax(prob)) for prob in y_proba])
    y_proba = model.predict_proba(X_test)
    print("predict classifier:\n",[int(np.argmax(prob)) for prob in y_proba])
    print("ytest:")
    print("", y_test.tolist())

    print("DONE")


if __name__ == "__main__":
    main()
