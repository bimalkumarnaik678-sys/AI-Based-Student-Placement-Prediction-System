"""
Feature logic shared by train_model.py and app.py.

Keeping it in one place guarantees that the website builds exactly the same
features the model was trained on.
"""

import pandas as pd

TARGET = "placement_status"

# Text columns the model treats as categories.
CATEGORICAL_FEATURES = ["gender", "branch", "college_tier", "volunteer_experience"]

# Columns that must never be used as inputs:
#  - student_id            -> just a row number
#  - salary_package_lpa    -> only exists AFTER a student is placed (target leakage)
ID_COLUMNS = ["student_id"]
LEAKAGE_COLUMNS = ["salary_package_lpa"]

# Cap used to turn the open-ended experience score into a 0-100 indicator.
EXPERIENCE_CAP = 60


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the combined score columns used by the model."""
    df = df.copy()

    df["academic_score"] = df["cgpa"] / 10 * 100

    df["technical_score"] = (
        df["coding_skill_score"] * 0.30
        + df["logical_reasoning_score"] * 0.20
        + df["aptitude_score"] * 0.20
        + df["mock_interview_score"] * 0.20
        + df["communication_skill_score"] * 0.10
    )

    df["experience_score"] = (
        df["internships_count"] * 5
        + df["projects_count"] * 3
        + df["certifications_count"] * 2
        + df["hackathons_participated"] * 2
        + df["github_repos"] * 1
    )

    df["professional_score"] = (
        df["leadership_score"] * 0.35
        + df["communication_skill_score"] * 0.35
        + df["extracurricular_score"] * 0.30
    )

    df["backlog_penalty"] = df["backlogs"] * 5

    df["placement_readiness"] = (
        df["academic_score"] * 0.20
        + df["technical_score"] * 0.40
        + df["experience_score"] * 0.20
        + df["professional_score"] * 0.20
        - df["backlog_penalty"]
    )

    return df
