from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
