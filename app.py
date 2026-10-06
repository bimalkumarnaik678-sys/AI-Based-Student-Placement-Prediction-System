"""
PlacePredict AI - Flask web app.

Routes
    GET  /                    Main student profile form
    POST /predict             Placement prediction page
    POST /resume              Resume analysis page
    POST /api/predict         JSON placement prediction API
    GET  /health              Hosting health check

Assessment System
    POST /start-assessment    Start interdisciplinary assessment
    GET  /assessment/<skill>  Generate skill-specific assessment
    POST /submit-assessment   Score assessment
    GET  /dashboard           Personalized assessment dashboard
"""

import json
import math
import os
import re

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    redirect,
    url_for,
)

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from features import (
    EXPERIENCE_CAP,
    LEAKAGE_COLUMNS,
    add_engineered_features,
)

# ---------------------------------------------------------------------
# NEW ASSESSMENT SYSTEM
# ---------------------------------------------------------------------

from assessment.engine import AssessmentEngine

from recommendation.roadmap import generate_roadmap


# =============================================================================
# APPLICATION SETUP
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# Required for Flask session
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "placement-ai-secret-key-change-this"
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# =============================================================================
# ASSESSMENT ENGINE
# =============================================================================

assessment_engine = AssessmentEngine()


# =============================================================================
# LOAD MODEL
# =============================================================================

model = None
feature_columns = []
categorical_features = []
model_metrics = {}
MODEL_ERROR = None


def load_model():

    global model
    global feature_columns
    global categorical_features
    global model_metrics
    global MODEL_ERROR

    try:

        model = joblib.load(
            os.path.join(
                MODEL_DIR,
                "placement_model.pkl"
            )
        )

        feature_columns = joblib.load(
            os.path.join(
                MODEL_DIR,
                "feature_columns.pkl"
            )
        )

        categorical_features = joblib.load(
            os.path.join(
                MODEL_DIR,
                "categorical_features.pkl"
            )
        )

    except FileNotFoundError as exc:

        MODEL_ERROR = (
            f"Model file not found "
            f"({os.path.basename(exc.filename or '')}). "
            "Run `python train_model.py` and copy the "
            "generated files into the model/ folder."
        )

        return

    except Exception as exc:

        MODEL_ERROR = (
            f"The model could not be loaded: {exc}"
        )

        return

    # ---------------------------------------------------------------
    # Leakage protection
    # ---------------------------------------------------------------

    leaked = [
        c
        for c in feature_columns
        if c in LEAKAGE_COLUMNS
    ]

    if leaked:

        model = None

        MODEL_ERROR = (
            "This model was trained with "
            + ", ".join(leaked)
            + " as an input. "
            "That value only exists after a student is placed, "
            "so visitors cannot provide it. "
            "Run `python train_model.py` to create a corrected model."
        )

        return

    # ---------------------------------------------------------------
    # Load model metrics
    # ---------------------------------------------------------------

    metrics_path = os.path.join(
        MODEL_DIR,
        "model_metrics.json"
    )

    if os.path.exists(metrics_path):

        try:

            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as fh:

                model_metrics = json.load(fh)

        except Exception:

            model_metrics = {}


load_model()


# =============================================================================
# FORM OPTIONS
# =============================================================================

GENDER_OPTIONS = [
    ("Male", "Male"),
    ("Female", "Female"),
    ("Other", "Other / prefer not to say"),
]


BRANCH_OPTIONS = [

    ("CSE", "Computer Science (CSE)"),
    ("IT", "Information Technology (IT)"),
    ("ECE", "Electronics & Communication (ECE)"),
    ("EEE", "Electrical & Electronics (EEE)"),
    ("Mechanical", "Mechanical"),
    ("Civil", "Civil"),

]


TIER_OPTIONS = [

    ("Tier 1", "Tier 1 (top institutes)"),
    ("Tier 2", "Tier 2"),
    ("Tier 3", "Tier 3"),

]


YES_NO = [
    ("Yes", "Yes"),
    ("No", "No"),
]


# =============================================================================
# SLIDER HELPER
# =============================================================================

def _slider(
    name,
    label,
    default,
    help_text
):

    return {

        "name": name,
        "label": label,
        "kind": "slider",
        "type": "float",

        "min": 0,
        "max": 100,
        "step": 1,

        "default": default,

        "help": help_text,

        "unit": "/100",

    }


# =============================================================================
# FORM FIELD GROUPS
# =============================================================================

FIELD_GROUPS = [

    {
        "id": "academics",

        "title": "Academics",

        "intro": "Your background and marks.",

        "fields": [

            {
                "name": "age",
                "label": "Age",
                "kind": "number",
                "type": "int",
                "min": 17,
                "max": 45,
                "step": 1,
                "default": 22,
                "help": "In years.",
                "unit": "",
            },

            {
                "name": "gender",
                "label": "Gender",
                "kind": "select",
                "type": "str",
                "options": GENDER_OPTIONS,
                "default": "Male",
                "help": "",
                "unit": "",
            },

            {
                "name": "branch",
                "label": "Branch",
                "kind": "select",
                "type": "str",
                "options": BRANCH_OPTIONS,
                "default": "CSE",
                "help": "",
                "unit": "",
            },

            {
                "name": "college_tier",
                "label": "College tier",
                "kind": "select",
                "type": "str",
                "options": TIER_OPTIONS,
                "default": "Tier 2",
                "help": "",
                "unit": "",
            },

            {
                "name": "cgpa",
                "label": "CGPA",
                "kind": "number",
                "type": "float",
                "min": 0,
                "max": 10,
                "step": 0.01,
                "default": 7.5,
                "help": "Out of 10.",
                "unit": "/10",
            },

            {
                "name": "attendance_percentage",
                "label": "Attendance",
                "kind": "number",
                "type": "float",
                "min": 0,
                "max": 100,
                "step": 0.1,
                "default": 80,
                "help": "Percent of classes attended.",
                "unit": "%",
            },

            {
                "name": "backlogs",
                "label": "Backlogs",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 20,
                "step": 1,
                "default": 0,
                "help": "Subjects not yet cleared.",
                "unit": "",
            },

        ],
    },

    {
        "id": "skills",

        "title": "Skills",

        "intro": (
            "Rate yourself using recent test or mock scores. "
            "Be honest, the estimate is only as good as these numbers."
        ),

        "fields": [

            _slider(
                "coding_skill_score",
                "Coding",
                60,
                "Coding tests, online judges, assessments."
            ),

            _slider(
                "aptitude_score",
                "Aptitude",
                60,
                "Quantitative and verbal test scores."
            ),

            _slider(
                "logical_reasoning_score",
                "Logical reasoning",
                60,
                "Puzzles and reasoning tests."
            ),

            _slider(
                "communication_skill_score",
                "Communication",
                60,
                "Speaking and writing in English."
            ),

            _slider(
                "mock_interview_score",
                "Mock interviews",
                60,
                "Average score from practice interviews."
            ),

            _slider(
                "leadership_score",
                "Leadership",
                55,
                "Clubs, teams, events you led."
            ),

            _slider(
                "extracurricular_score",
                "Extracurricular",
                55,
                "Sports, arts, societies, competitions."
            ),

        ],
    },

    {
        "id": "experience",

        "title": "Experience",

        "intro": (
            "What you have built and done outside the classroom."
        ),

        "fields": [

            {
                "name": "internships_count",
                "label": "Internships",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 10,
                "step": 1,
                "default": 1,
                "help": "Completed internships.",
                "unit": "",
            },

            {
                "name": "projects_count",
                "label": "Projects",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 30,
                "step": 1,
                "default": 3,
                "help": "Academic and personal projects.",
                "unit": "",
            },

            {
                "name": "certifications_count",
                "label": "Certifications",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 30,
                "step": 1,
                "default": 2,
                "help": "Completed courses with a certificate.",
                "unit": "",
            },

            {
                "name": "hackathons_participated",
                "label": "Hackathons",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 30,
                "step": 1,
                "default": 1,
                "help": "Events you took part in.",
                "unit": "",
            },

            {
                "name": "github_repos",
                "label": "GitHub repositories",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 200,
                "step": 1,
                "default": 5,
                "help": "Public repositories.",
                "unit": "",
            },

            {
                "name": "linkedin_connections",
                "label": "LinkedIn connections",
                "kind": "number",
                "type": "int",
                "min": 0,
                "max": 30000,
                "step": 1,
                "default": 300,
                "help": "Approximate count.",
                "unit": "",
            },

            {
                "name": "volunteer_experience",
                "label": "Volunteer experience",
                "kind": "select",
                "type": "str",
                "options": YES_NO,
                "default": "No",
                "help": "",
                "unit": "",
            },

        ],
    },

    {
        "id": "routine",

        "title": "Daily routine",

        "intro": "Typical day during the semester.",

        "fields": [

            {
                "name": "study_hours_per_day",
                "label": "Study hours per day",
                "kind": "number",
                "type": "float",
                "min": 0,
                "max": 16,
                "step": 0.5,
                "default": 4,
                "help": "Outside class hours.",
                "unit": "hrs",
            },

            {
                "name": "sleep_hours",
                "label": "Sleep hours per day",
                "kind": "number",
                "type": "float",
                "min": 0,
                "max": 14,
                "step": 0.5,
                "default": 7,
                "help": "Average per night.",
                "unit": "hrs",
            },

        ],
    },

]


ALL_FIELDS = [
    field
    for group in FIELD_GROUPS
    for field in group["fields"]
]


FIELD_BY_NAME = {
    field["name"]: field
    for field in ALL_FIELDS
}


def default_values():

    return {
        field["name"]: field["default"]
        for field in ALL_FIELDS
    }


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def parse_student(source):

    values = {}
    errors = {}

    for field in ALL_FIELDS:

        name = field["name"]
        label = field["label"]

        raw = source.get(name)

        raw = (
            ""
            if raw is None
            else str(raw).strip()
        )

        if raw == "":

            errors[name] = (
                f"Enter a value for {label.lower()}."
            )

            continue

        if field["kind"] == "select":

            allowed = [
                value
                for value, _
                in field["options"]
            ]

            if raw not in allowed:

                errors[name] = (
                    f"Choose one of the listed options "
                    f"for {label.lower()}."
                )

            else:

                values[name] = raw

            continue

        try:

            number = float(raw)

        except ValueError:

            errors[name] = (
                f"{label} must be a number."
            )

            continue

        if (
            math.isnan(number)
            or math.isinf(number)
        ):

            errors[name] = (
                f"{label} must be a number."
            )

            continue

        if (
            number < field["min"]
            or number > field["max"]
        ):

            errors[name] = (
                f"{label} must be between "
                f"{field['min']} and {field['max']}."
            )

            continue

        if field["type"] == "int":

            if not float(number).is_integer():

                errors[name] = (
                    f"{label} must be a whole number."
                )

                continue

            number = int(number)

        values[name] = number

    return values, errors


# =============================================================================
# PLACEMENT PREDICTION
# =============================================================================

def predict_placement(values):

    row = dict(values)

    if row.get("gender") not in (
        "Male",
        "Female"
    ):

        row["gender"] = "Unknown"

    df = add_engineered_features(
        pd.DataFrame([row])
    )

    missing = [
        c
        for c in feature_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "The model expects inputs this form "
            "does not collect: "
            + ", ".join(missing)
        )

    df = df.reindex(
        columns=feature_columns
    )

    for col in categorical_features:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    numeric_cols = [
        c
        for c in df.columns
        if c not in categorical_features
    ]

    df[numeric_cols] = (
        df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    probabilities = model.predict_proba(df)[0]

    placed_index = list(
        model.classes_
    ).index(1)

    placed = (
        float(probabilities[placed_index])
        * 100
    )

    rounded_placed = round(
        placed,
        1
    )

    return {

        "placed_probability":
            rounded_placed,

        "not_placed_probability":
            round(
                100 - rounded_placed,
                1
            ),

    }


# =============================================================================
# PROFILE INDICATORS
# =============================================================================

def profile_indicators(v):

    academic = v["cgpa"] * 10

    technical = (
        v["coding_skill_score"] * 0.30
        + v["logical_reasoning_score"] * 0.20
        + v["aptitude_score"] * 0.20
        + v["mock_interview_score"] * 0.20
        + v["communication_skill_score"] * 0.10
    )

    experience_raw = (
        v["internships_count"] * 5
        + v["projects_count"] * 3
        + v["certifications_count"] * 2
        + v["hackathons_participated"] * 2
        + v["github_repos"]
    )

    experience = min(
        100,
        experience_raw
        / EXPERIENCE_CAP
        * 100
    )

    professional = (
        v["leadership_score"] * 0.35
        + v["communication_skill_score"] * 0.35
        + v["extracurricular_score"] * 0.30
    )

    overall = (
        academic * 0.2
        + technical * 0.4
        + experience * 0.2
        + professional * 0.2
    )

    overall = max(
        0,
        min(
            100,
            overall - v["backlogs"] * 5
        )
    )

    indicators = [

        {
            "label": "Academics",
            "value": round(academic)
        },

        {
            "label": "Technical skills",
            "value": round(technical)
        },

        {
            "label": "Experience",
            "value": round(experience)
        },

        {
            "label": "Professional skills",
            "value": round(professional)
        },

    ]

    return indicators, round(overall)


# =============================================================================
# EXISTING INSIGHTS
# =============================================================================

def build_insights(v, indicators):

    good = []
    improve = []

    def add(
        bucket,
        title,
        text,
        weight
    ):

        bucket.append(
            {
                "title": title,
                "text": text,
                "weight": weight,
            }
        )

    # Academics

    if v["cgpa"] >= 8:

        add(
            good,
            "Strong CGPA",
            f"A CGPA of {v['cgpa']:.2f} "
            "is a positive academic indicator.",
            3,
        )

    elif v["cgpa"] < 6.5:

        add(
            improve,
            "Raise your CGPA",
            "Focus on scoring well in the remaining semesters.",
            3,
        )

    elif v["cgpa"] < 7:

        add(
            improve,
            "CGPA can be improved",
            "Improving your CGPA may expand the number of opportunities "
            "for which you meet academic criteria.",
            2,
        )

    if v["backlogs"] > 0:

        add(
            improve,
            "Clear your backlogs",
            f"You currently have {v['backlogs']} pending subject(s).",
            4,
        )

    else:

        add(
            good,
            "No backlogs",
            "You currently have no pending academic backlogs.",
            2,
        )

    if v["attendance_percentage"] < 75:

        add(
            improve,
            "Improve attendance",
            "Your attendance is below 75%.",
            2,
        )

    # Skills

    if v["coding_skill_score"] >= 75:

        add(
            good,
            "Strong coding",
            "Your coding assessment score is currently strong.",
            3,
        )

    elif v["coding_skill_score"] < 70:

        add(
            improve,
            "Improve coding",
            "Use structured practice covering arrays, strings, "
            "hashing, recursion and other core topics.",
            4,
        )

    if (
        v["aptitude_score"] < 65
        or v["logical_reasoning_score"] < 65
    ):

        add(
            improve,
            "Build aptitude and reasoning",
            "Use timed quantitative and reasoning practice tests.",
            3,
        )

    elif (
        v["aptitude_score"] >= 75
        and v["logical_reasoning_score"] >= 75
    ):

        add(
            good,
            "Strong aptitude and reasoning",
            "Your current aptitude and reasoning scores are strong.",
            2,
        )

    if (
        v["communication_skill_score"] < 65
        or v["mock_interview_score"] < 65
    ):

        add(
            improve,
            "Improve interview preparation",
            "Practice interview questions and record mock responses.",
            3,
        )

    elif v["mock_interview_score"] >= 75:

        add(
            good,
            "Strong mock interview score",
            "Your current mock interview score is strong.",
            2,
        )

    # Experience

    if v["internships_count"] == 0:

        add(
            improve,
            "Gain practical experience",
            "Consider internships, live projects or supervised practical work.",
            4,
        )

    elif v["internships_count"] >= 2:

        add(
            good,
            "Internship experience",
            f"You have completed {v['internships_count']} internships.",
            3,
        )

    if v["projects_count"] < 2:

        add(
            improve,
            "Build more projects",
            "Build projects that demonstrate practical skills.",
            3,
        )

    elif v["projects_count"] >= 4:

        add(
            good,
            "Good project portfolio",
            f"You currently list {v['projects_count']} projects.",
            2,
        )

    if v["github_repos"] < 3:

        add(
            improve,
            "Publish projects on GitHub",
            "Publish relevant projects with clear README files.",
            1,
        )

    if v["certifications_count"] == 0:

        add(
            improve,
            "Add relevant certification",
            "Consider a certification aligned with your career track.",
            1,
        )

    if v["hackathons_participated"] >= 2:

        add(
            good,
            "Hackathon experience",
            "Hackathons provide additional practical experience.",
            1,
        )

    if len(improve) < 2:

        weakest = min(
            indicators,
            key=lambda item: item["value"]
        )

        advice = {

            "Academics":
                "Maintain consistent semester performance.",

            "Technical skills":
                "Combine practical technical projects with structured skill practice.",

            "Experience":
                "Add practical projects, internships or relevant activities.",

            "Professional skills":
                "Practice communication, presentation and teamwork.",

        }

        add(
            improve,
            f"Develop {weakest['label'].lower()}",
            advice[weakest["label"]],
            1,
        )

    good.sort(
        key=lambda item: -item["weight"]
    )

    improve.sort(
        key=lambda item: -item["weight"]
    )

    return (
        good[:4],
        improve[:4]
    )


# =============================================================================
# PLACEMENT CHANCE DESCRIPTION
# =============================================================================

def describe_chance(placed):

    if placed >= 65:
        return "Strong chance", "high"

    if placed >= 55:
        return "Good chance", "good"

    if placed >= 45:
        return "Around even", "mid"

    if placed >= 35:
        return "Below even", "low"

    return "Low chance", "poor"


# =============================================================================
# INPUT SUMMARY
# =============================================================================

def summarise_inputs(values):

    rows = []

    for field in ALL_FIELDS:

        raw = values[field["name"]]

        if field["kind"] == "select":

            display = dict(
                field["options"]
            ).get(
                raw,
                raw
            )

        else:

            display = (
                f"{raw:g}"
                if isinstance(raw, float)
                else str(raw)
            )

            if field["unit"]:

                display += (
                    f" {field['unit']}"
                    if field["unit"] == "hrs"
                    else field["unit"]
                )

        rows.append(
            {
                "label": field["label"],
                "value": display,
            }
        )

    return rows


# =============================================================================
# HOME
# =============================================================================

@app.route("/")
def home():

    return render_template(

        "index.html",

        groups=FIELD_GROUPS,

        values=default_values(),

        errors={},

        metrics=model_metrics,

        model_error=MODEL_ERROR,

        experience_cap=EXPERIENCE_CAP,

    )


# =============================================================================
# PLACEMENT PREDICTION
# =============================================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    if request.method == "GET":

        return render_template(

            "index.html",

            groups=FIELD_GROUPS,

            values=default_values(),

            errors={},

            metrics=model_metrics,

            model_error=MODEL_ERROR,

            experience_cap=EXPERIENCE_CAP,

        )

    if MODEL_ERROR:

        return render_template(
            "error.html",
            title="Model not ready",
            message=MODEL_ERROR
        ), 503

    values, errors = parse_student(
        request.form
    )

    if errors:

        merged = default_values()

        merged.update(
            {
                k: request.form.get(
                    k,
                    merged[k]
                )
                for k in merged
            }
        )

        return render_template(

            "index.html",

            groups=FIELD_GROUPS,

            values=merged,

            errors=errors,

            metrics=model_metrics,

            model_error=MODEL_ERROR,

            experience_cap=EXPERIENCE_CAP,

        ), 400

    try:

        result = predict_placement(
            values
        )

    except Exception as exc:

        app.logger.exception(
            "Prediction failed"
        )

        return render_template(

            "error.html",

            title="Prediction failed",

            message=str(exc),

        ), 500

    indicators, overall = profile_indicators(
        values
    )

    strengths, improvements = build_insights(
        values,
        indicators
    )

    verdict, band = describe_chance(
        result["placed_probability"]
    )

    return render_template(

        "result.html",

        placed=result["placed_probability"],

        not_placed=result["not_placed_probability"],

        verdict=verdict,

        band=band,

        indicators=indicators,

        overall=overall,

        strengths=strengths,

        improvements=improvements,

        inputs=summarise_inputs(values),

        metrics=model_metrics,

    )


# =============================================================================
# JSON PREDICTION API
# =============================================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    if MODEL_ERROR:

        return jsonify(
            {
                "error": MODEL_ERROR
            }
        ), 503

    payload = request.get_json(
        silent=True
    )

    if not isinstance(payload, dict):

        return jsonify(
            {
                "error": "Send a JSON object."
            }
        ), 400

    values, errors = parse_student(
        payload
    )

    if errors:

        return jsonify(
            {
                "errors": errors
            }
        ), 400

    try:

        result = predict_placement(
            values
        )

    except Exception as exc:

        app.logger.exception(
            "API prediction failed"
        )

        return jsonify(
            {
                "error": str(exc)
            }
        ), 500

    verdict, _ = describe_chance(
        result["placed_probability"]
    )

    return jsonify(
        {
            **result,
            "verdict": verdict
        }
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/health")
def health():

    return jsonify(
        {
            "status":
                "ok"
                if not MODEL_ERROR
                else "model_missing"
        }
    ), (
        200
        if not MODEL_ERROR
        else 503
    )


# =============================================================================
# INTERDISCIPLINARY ASSESSMENT SYSTEM
# =============================================================================

DISCIPLINE_OPTIONS = [
    "MCA", "MBA", "B.Tech", "BBA", "BCA", "CSE", "IT", "ECE", "Other",
]

CAREER_INTEREST_OPTIONS = [
    "Software / IT", "Data / AI", "Cybersecurity", "Web Development",
    "Data Analytics", "Management", "Marketing", "Finance", "Other",
]

# These keys MUST match the "skill" field in the question-bank JSON files.
ASSESSMENT_SKILLS = [
    {"key": "aptitude", "label": "Aptitude", "description": "Quantitative and logical maths problems."},
    {"key": "reasoning", "label": "Reasoning", "description": "Series, analogies and logical deduction."},
    {"key": "programming", "label": "Programming", "description": "Technical coding and programming fundamentals."},
    {"key": "dbms", "label": "DBMS", "description": "SQL, normalization and database concepts."},
    {"key": "communication", "label": "Communication", "description": "Professional and interview communication."},
    {"key": "leadership", "label": "Leadership", "description": "Team leadership and decision-making scenarios."},
    {"key": "problem_solving", "label": "Problem Solving", "description": "Breaking down and solving real-world problems."},
    {"key": "marketing", "label": "Marketing", "description": "Core marketing and brand strategy concepts."},
]


def _assessment_skill_label(skill):
    """Return the display label for an assessment skill."""

    return next(
        (
            item["label"]
            for item in ASSESSMENT_SKILLS
            if item["key"] == skill
        ),
        skill.replace("_", " ").title(),
    )


def _validate_question_bank(questions):
    """
    Validate the five questions before they are displayed.

    The project now uses ONE answer convention everywhere:

        0 -> first option
        1 -> second option
        2 -> third option
        3 -> fourth option

    We intentionally do not silently convert JSON answer keys. A wrong
    answer key in a JSON file must be corrected in that JSON file instead of
    being guessed by the application.
    """

    validated = []

    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question {position} is not a JSON object."
            )

        question_id = question.get("id")
        question_text = str(question.get("question", "")).strip()
        options = question.get("options")
        answer = question.get("answer")

        if question_id is None:
            raise ValueError(
                f"Question {position} is missing its 'id'."
            )

        if not question_text:
            raise ValueError(
                f"Question {question_id} has no question text."
            )

        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(
                f"Question {question_id} must have exactly 4 options."
            )

        try:
            answer = int(answer)
        except (TypeError, ValueError):
            raise ValueError(
                f"Question {question_id} has an invalid 'answer'. "
                "Use 0, 1, 2 or 3."
            )

        if answer < 0 or answer >= len(options):
            raise ValueError(
                f"Question {question_id} has answer={answer}. "
                "With the current 0-based format, answer must be 0, 1, 2 or 3."
            )

        # Make a shallow copy so the assessment route never modifies the
        # original object held by the question-bank engine.
        clean_question = dict(question)
        clean_question["id"] = question_id
        clean_question["options"] = [str(option) for option in options]
        clean_question["answer"] = answer

        validated.append(clean_question)

    return validated


def _score_assessment(questions, form_data):
    """
    Score an assessment using the SAME 0-based convention as the JSON files.

    New assessment.html submits:

        q_<question_id> = 0/1/2/3

    For example, if question id is 801 and the user selects the first
    option, the browser sends:

        q_801=0

    A temporary compatibility path also understands the old form format
    q_1, q_2, ... with values 1, 2, 3, 4. This compatibility is based on the
    field name, not on guessing from the answer value, so it cannot silently
    shift a legitimate 0-based answer.
    """

    correct = 0
    wrong = 0
    unanswered = 0
    details = []

    for position, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        options = question["options"]
        correct_index = int(question["answer"])

        # -----------------------------------------------------------
        # PRIMARY FORMAT: q_<question_id>
        # Value is 0-based: 0, 1, 2, 3.
        # -----------------------------------------------------------
        id_key = f"q_{question_id}"
        selected_raw = form_data.get(id_key)
        source = "question_id"

        # -----------------------------------------------------------
        # LEGACY COMPATIBILITY: q_1, q_2, ...
        # Old template used 1-based values 1, 2, 3, 4.
        # Convert ONLY this legacy format to 0-based.
        # -----------------------------------------------------------
        if selected_raw is None:
            position_key = f"q_{position}"
            selected_raw = form_data.get(position_key)
            source = "question_position"

            if selected_raw not in (None, ""):
                try:
                    selected_raw = int(selected_raw) - 1
                except (TypeError, ValueError):
                    selected_raw = None

        else:
            try:
                selected_raw = int(selected_raw)
            except (TypeError, ValueError):
                selected_raw = None

        selected_index = selected_raw

        # -----------------------------------------------------------
        # Validate submitted option index.
        # -----------------------------------------------------------
        if selected_index is None:
            unanswered += 1
            is_correct = False
        elif not (0 <= selected_index < len(options)):
            wrong += 1
            is_correct = False
        elif selected_index == correct_index:
            correct += 1
            is_correct = True
        else:
            wrong += 1
            is_correct = False

        selected_text = None
        if (
            selected_index is not None
            and 0 <= selected_index < len(options)
        ):
            selected_text = options[selected_index]

        correct_text = options[correct_index]

        details.append({
            "id": question_id,
            "question": question.get("question", ""),
            "options": options,
            "selected_index": selected_index,
            "selected_text": selected_text,
            "correct_index": correct_index,
            "correct_text": correct_text,
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
            "answer_source": source if selected_raw is not None else None,
        })

    total = len(questions)
    percentage = round((correct / total) * 100, 2) if total else 0

    return {
        "score": correct,
        "correct": correct,
        "wrong": wrong,
        "unanswered": unanswered,
        "total": total,
        "percentage": percentage,
        "details": details,
    }


@app.route("/start-assessment", methods=["GET", "POST"])
def start_assessment():
    """Start a fresh interdisciplinary assessment session."""

    if request.method == "GET":
        return render_template(
            "start_assessment.html",
            discipline_options=DISCIPLINE_OPTIONS,
            career_options=CAREER_INTEREST_OPTIONS,
        )

    discipline = request.form.get("discipline") or "ALL"
    career_interest = request.form.get("career_interest") or "Other"

    session["discipline"] = discipline
    session["career_interest"] = career_interest
    session["assessment_scores"] = {}

    session.pop("current_questions", None)
    session.pop("current_skill", None)

    return redirect(url_for("assessment_menu"))


# =============================================================================
# ASSESSMENT MENU
# =============================================================================

@app.route("/assessment-menu")
def assessment_menu():
    discipline = session.get("discipline", "ALL")
    career_interest = session.get("career_interest", "Other")
    assessment_scores = session.get("assessment_scores", {})

    return render_template(
        "assessment_menu.html",
        discipline=discipline,
        career_interest=career_interest,
        assessment_scores=assessment_scores,
        skills=ASSESSMENT_SKILLS,
    )


# =============================================================================
# GENERATE SKILL ASSESSMENT
# =============================================================================

@app.route("/assessment/<skill>")
def assessment(skill):
    """Generate exactly five questions for one assessment skill."""

    if skill not in {item["key"] for item in ASSESSMENT_SKILLS}:
        return render_template(
            "error.html",
            title="Assessment unavailable",
            message=f"Unknown assessment skill: '{skill}'.",
        ), 404

    discipline = session.get("discipline", "ALL")
    career_interest = session.get("career_interest", "Other")

    try:
        questions = assessment_engine.generate_test(
            discipline=discipline,
            career_interest=career_interest,
            skill=skill,
            number_of_questions=5,
        )

        questions = _validate_question_bank(questions)

    except Exception as exc:
        app.logger.exception(
            "Assessment generation/validation failed for skill=%s",
            skill,
        )
        return render_template(
            "error.html",
            title="Assessment unavailable",
            message=f"Could not generate the {skill} assessment: {exc}",
        ), 500

    if not questions:
        return render_template(
            "error.html",
            title="Assessment unavailable",
            message=(
                f"No questions are currently available for '{skill}' "
                "with the selected discipline and career track."
            ),
        ), 404

    # Keep the exact generated questions for this attempt. The scorer will
    # compare the submitted option index with these same question objects.
    session["current_questions"] = questions
    session["current_skill"] = skill

    app.logger.info(
        "Assessment started: skill=%s questions=%s ids=%s",
        skill,
        len(questions),
        [str(q["id"]) for q in questions],
    )

    return render_template(
        "assessment.html",
        questions=questions,
        skill=skill,
        skill_label=_assessment_skill_label(skill),
        discipline=discipline,
        career_interest=career_interest,
    )


# =============================================================================
# SUBMIT ASSESSMENT
# =============================================================================

@app.route("/submit-assessment", methods=["POST"])
def submit_assessment():
    """Receive the radio-button input and score the exact active test."""

    questions = session.get("current_questions", [])
    skill = session.get("current_skill")

    if not questions or not skill:
        app.logger.warning(
            "Assessment submission rejected: session has no active questions."
        )
        return render_template(
            "error.html",
            title="Assessment session expired",
            message=(
                "Your assessment session is no longer active. "
                "Please start the assessment again."
            ),
        ), 400

    try:
        questions = _validate_question_bank(questions)
    except Exception as exc:
        app.logger.exception("Stored assessment question validation failed")
        return render_template(
            "error.html",
            title="Assessment data error",
            message=str(exc),
        ), 500

    # Flask receives radio buttons through request.form.
    form_data = request.form.to_dict(flat=True)

    app.logger.info(
        "Assessment submitted: skill=%s form_keys=%s form_values=%s",
        skill,
        list(form_data.keys()),
        form_data,
    )

    result = _score_assessment(
        questions,
        form_data,
    )

    app.logger.info(
        "Assessment result: skill=%s correct=%s wrong=%s unanswered=%s percentage=%s",
        skill,
        result["correct"],
        result["wrong"],
        result["unanswered"],
        result["percentage"],
    )

    scores = dict(session.get("assessment_scores", {}))
    scores[skill] = result["percentage"]
    session["assessment_scores"] = scores

    # Clear only the active test. Previous completed skill scores remain.
    session.pop("current_questions", None)
    session.pop("current_skill", None)

    return render_template(
        "assessment_result.html",
        result=result,
        skill=skill,
        skill_label=_assessment_skill_label(skill),
        scores=scores,
    )


# =============================================================================
# ASSESSMENT DASHBOARD
# =============================================================================

@app.route("/dashboard")
def dashboard():
    assessment_scores = session.get("assessment_scores", {})
    discipline = session.get("discipline", "ALL")
    career_interest = session.get("career_interest", "Other")

    # recommendation.engine does not expose a compatible recommendation
    # function in the current project, so keep this list empty.
    recommendations = []

    roadmap = generate_roadmap(
        assessment_scores
    )

    if assessment_scores:
        placement_readiness = round(
            sum(float(score) for score in assessment_scores.values())
            / len(assessment_scores)
        )
    else:
        placement_readiness = 0

    return render_template(
        "dashboard.html",
        assessment_scores=assessment_scores,
        recommendations=recommendations,
        roadmap=roadmap,
        discipline=discipline,
        career_interest=career_interest,
        placement_readiness=placement_readiness,
        skills=ASSESSMENT_SKILLS,
    )

# =============================================================================
# RESUME ANALYSIS
# =============================================================================

def _skill(
    name,
    *aliases,
    case_sensitive=False
):

    return {

        "name": name,

        "patterns": [
            name,
            *aliases
        ],

        "case_sensitive":
            case_sensitive,

    }


SKILL_CATEGORIES = {

    "Programming languages": [

        _skill("Python"),
        _skill("Java"),
        _skill("C++"),
        _skill("C", case_sensitive=True),
        _skill("C#"),
        _skill("JavaScript"),
        _skill("TypeScript"),
        _skill("PHP"),
        _skill("Kotlin"),
        _skill("Swift"),

    ],

    "Web & backend": [

        _skill("HTML"),
        _skill("CSS"),
        _skill(
            "React",
            "React.js",
            "ReactJS"
        ),
        _skill(
            "Node.js",
            "NodeJS",
            "Node js"
        ),
        _skill("Angular"),
        _skill("Flask"),
        _skill("Django"),
        _skill("FastAPI"),
        _skill("Spring Boot"),
        _skill(
            "REST API",
            "RESTful"
        ),
        _skill("Bootstrap"),
        _skill("Tailwind"),

    ],

    "Databases": [

        _skill("SQL"),
        _skill("MySQL"),
        _skill(
            "PostgreSQL",
            "Postgres"
        ),
        _skill("MongoDB"),
        _skill("Oracle"),
        _skill("SQLite"),
        _skill("Redis"),
        _skill("Firebase"),

    ],

    "Data & AI": [

        _skill("Machine Learning"),
        _skill("Deep Learning"),
        _skill(
            "NLP",
            "Natural Language Processing"
        ),
        _skill("Computer Vision"),
        _skill("Data Science"),
        _skill(
            "Data Analysis",
            "Data Analytics"
        ),
        _skill("TensorFlow"),
        _skill("PyTorch"),
        _skill(
            "Scikit-learn",
            "sklearn"
        ),
        _skill("Pandas"),
        _skill("NumPy"),
        _skill(
            "Power BI",
            "PowerBI"
        ),
        _skill("Tableau"),
        _skill("Excel"),
        _skill("Statistics"),

    ],

    "Cloud & tools": [

        _skill(
            "AWS",
            "Amazon Web Services"
        ),
        _skill("Azure"),
        _skill(
            "GCP",
            "Google Cloud"
        ),
        _skill("Docker"),
        _skill("Kubernetes"),
        _skill("Git"),
        _skill("GitHub"),
        _skill("Linux"),
        _skill("CI/CD"),

    ],

    "Computer science core": [

        _skill("Data Structures"),
        _skill("Algorithms"),
        _skill("DSA"),
        _skill(
            "OOP",
            "Object Oriented",
            "Object-Oriented"
        ),
        _skill("DBMS"),
        _skill("Operating Systems"),
        _skill("Computer Networks"),
        _skill("System Design"),

    ],

    "Professional skills": [

        _skill("Communication"),
        _skill("Leadership"),
        _skill("Teamwork"),
        _skill(
            "Problem Solving",
            "Problem-Solving"
        ),
        _skill("Time Management"),
        _skill("Project Management"),
        _skill("Presentation"),

    ],

}


IN_DEMAND = [

    "Python",
    "SQL",
    "Git",
    "Data Structures",
    "Java",
    "REST API",
    "Docker",
    "AWS",
    "Communication",

]


SECTION_PATTERNS = {

    "Summary or objective":
        r"^(professional\s+)?"
        r"(summary|objective|profile|about\s+me|"
        r"career\s+objective)\b",

    "Education":
        r"^(education|academic|qualification)",

    "Skills":
        r"^(technical\s+)?"
        r"(skills|technologies|tech\s+stack|"
        r"core\s+competenc)",

    "Projects":
        r"^(academic\s+)?projects?\b",

    "Experience":
        r"^(work\s+|professional\s+)?"
        r"(experience|internships?|employment)\b",

    "Certifications":
        r"^(certifications?|courses|licen[cs]es)",

    "Achievements":
        r"^(achievements?|awards?|honou?rs|"
        r"accomplishments|extra[\s-]?curricular)",

}


def _compile_skill(entry):

    flags = (
        0
        if entry["case_sensitive"]
        else re.IGNORECASE
    )

    options = "|".join(
        re.escape(p)
        for p in entry["patterns"]
    )

    return re.compile(

        rf"(?<![\w+#])"
        rf"(?:{options})"
        rf"(?![\w+#])",

        flags

    )


_COMPILED_SKILLS = {

    category: [

        (
            entry["name"],
            _compile_skill(entry)
        )

        for entry in entries

    ]

    for category, entries
    in SKILL_CATEGORIES.items()

}


# =============================================================================
# PDF TEXT EXTRACTION
# =============================================================================

def extract_pdf_text(file_storage):

    header = file_storage.stream.read(5)

    file_storage.stream.seek(0)

    if header != b"%PDF-":

        raise ValueError(
            "That file is not a PDF. "
            "Upload your resume as a .pdf file."
        )

    try:

        reader = PdfReader(
            file_storage.stream
        )

        if reader.is_encrypted:

            if not reader.decrypt(""):

                raise ValueError(
                    "This PDF is password protected. "
                    "Remove the password and try again."
                )

        pages = [

            (
                page.extract_text()
                or ""
            )

            for page in reader.pages[:8]

        ]

    except PdfReadError:

        raise ValueError(
            "The PDF could not be read. "
            "Try exporting it again from your editor."
        )

    text = "\n".join(pages).strip()

    if len(text) < 80:

        raise ValueError(
            "No readable text was found. "
            "The PDF may be a scanned image. "
            "Export it as a text-based PDF from Word "
            "or Google Docs and try again."
        )

    return text


# =============================================================================
# RESUME ANALYSIS
# =============================================================================

def analyse_resume(text):

    found_by_category = {}
    found_names = set()

    for category, compiled in _COMPILED_SKILLS.items():

        hits = [

            name

            for name, pattern
            in compiled

            if pattern.search(text)

        ]

        if hits:

            found_by_category[
                category
            ] = hits

            found_names.update(hits)

    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip()
    ]

    sections = {}

    for label, pattern in SECTION_PATTERNS.items():

        regex = re.compile(
            pattern,
            re.IGNORECASE
        )

        sections[label] = any(

            len(ln) <= 45
            and regex.search(ln)

            for ln in lines

        )

    contact = {

        "Email":
            bool(
                re.search(
                    r"[\w.+-]+@[\w-]+\.[\w.-]+",
                    text
                )
            ),

        "Phone":
            bool(
                re.search(
                    r"(?<!\d)"
                    r"(?:\+?\d{1,3}[\s-]?)?"
                    r"\d{10}"
                    r"(?!\d)"
                    r"|\b\d{3,5}[\s-]\d{5,7}\b",
                    text
                )
            ),

        "LinkedIn":
            "linkedin.com"
            in text.lower(),

        "GitHub":
            "github.com"
            in text.lower(),

    }

    words = len(
        re.findall(
            r"\w+",
            text
        )
    )

    # ---------------------------------------------------------------
    # Resume score
    # ---------------------------------------------------------------

    skill_points = round(
        min(
            len(found_names),
            15
        )
        / 15
        * 40
    )

    section_points = round(
        sum(sections.values())
        / len(sections)
        * 30
    )

    contact_points = round(
        sum(contact.values())
        / len(contact)
        * 15
    )

    if 250 <= words <= 900:

        length_points = 15

    elif (
        150 <= words < 250
        or 900 < words <= 1200
    ):

        length_points = 8

    else:

        length_points = 3

    score = (
        skill_points
        + section_points
        + contact_points
        + length_points
    )

    suggestions = []

    missing_sections = [

        label

        for label, present
        in sections.items()

        if not present

    ]

    for label in (
        "Projects",
        "Skills",
        "Education",
        "Experience"
    ):

        if label in missing_sections:

            suggestions.append(

                f"Add a clearly titled {label} "
                "section so recruiters and screening "
                "software can find it."

            )

    if not contact["LinkedIn"]:

        suggestions.append(
            "Add your LinkedIn profile link."
        )

    if not contact["GitHub"]:

        suggestions.append(
            "Add your GitHub link if you have projects to show."
        )

    if (
        not contact["Email"]
        or not contact["Phone"]
    ):

        suggestions.append(
            "Put a working email address and phone number at the top."
        )

    if len(found_names) < 8:

        suggestions.append(
            "List more specific tools and technologies."
        )

    if words < 250:

        suggestions.append(
            "Your resume is quite short. "
            "Describe what you built and the result in each project."
        )

    elif words > 900:

        suggestions.append(
            "Your resume is long. "
            "Trim it to one page or two at most."
        )

    missing_in_demand = [

        skill

        for skill in IN_DEMAND

        if skill not in found_names

    ][:6]

    return {

        "score": score,

        "breakdown": [

            {
                "label": "Skills found",
                "points": skill_points,
                "out_of": 40,
            },

            {
                "label": "Standard sections",
                "points": section_points,
                "out_of": 30,
            },

            {
                "label": "Contact and links",
                "points": contact_points,
                "out_of": 15,
            },

            {
                "label": "Length",
                "points": length_points,
                "out_of": 15,
            },

        ],

        "skills": found_by_category,

        "skill_count":
            len(found_names),

        "sections":
            sections,

        "contact":
            contact,

        "words":
            words,

        "suggestions":
            suggestions[:6],

        "missing_in_demand":
            missing_in_demand,

    }


# =============================================================================
# RESUME ROUTE
# =============================================================================

@app.route(
    "/resume",
    methods=["POST"]
)
def resume():

    upload = request.files.get(
        "resume"
    )

    if (
        not upload
        or not upload.filename
    ):

        return render_template(

            "error.html",

            title="No file selected",

            message=(
                "Choose a PDF resume to analyse."
            ),

        ), 400

    try:

        text = extract_pdf_text(
            upload
        )

    except ValueError as exc:

        return render_template(

            "error.html",

            title="Could not read the resume",

            message=str(exc),

        ), 400

    except Exception:

        app.logger.exception(
            "Resume analysis failed"
        )

        return render_template(

            "error.html",

            title="Could not read the resume",

            message=(
                "Something went wrong while "
                "reading the PDF. Try a different file."
            ),

        ), 500

    analysis = analyse_resume(
        text
    )

    verdict = (

        "Strong resume"
        if analysis["score"] >= 75

        else "Good start"
        if analysis["score"] >= 50

        else "Needs work"

    )

    return render_template(

        "resume_result.html",

        a=analysis,

        verdict=verdict,

        filename=upload.filename,

    )


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def too_large(_):

    return render_template(

        "error.html",

        title="File too large",

        message=(
            "Resumes must be 5 MB or smaller."
        ),

    ), 413


@app.errorhandler(404)
def not_found(_):

    return render_template(

        "error.html",

        title="Page not found",

        message=(
            "That page does not exist."
        ),

    ), 404


@app.errorhandler(500)
def server_error(_):

    return render_template(

        "error.html",

        title="Something went wrong",

        message=(
            "An unexpected error occurred. "
            "Please try again."
        ),

    ), 500


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )

