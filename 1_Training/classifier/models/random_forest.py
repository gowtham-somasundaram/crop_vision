
from sklearn.ensemble import RandomForestClassifier


def build_model():
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )
