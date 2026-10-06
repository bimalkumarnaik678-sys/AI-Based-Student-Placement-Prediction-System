def generate_roadmap(assessment_scores):

    if not assessment_scores:
        return []

    roadmap = []

    # ---------------------------------------------------------
    # Find weakest areas
    # ---------------------------------------------------------

    sorted_skills = sorted(
        assessment_scores.items(),
        key=lambda item: item[1]
    )

    # Maximum 4 focus areas
    focus_skills = sorted_skills[:4]

    # ---------------------------------------------------------
    # Skill-specific learning content
    # ---------------------------------------------------------

    learning_content = {

        "reasoning": {
            "title": "Logical Reasoning",
            "goal": "Improve logical thinking and problem-solving speed.",
            "topics": [
                "Number series",
                "Coding-decoding",
                "Blood relations",
                "Direction sense",
                "Logical puzzles"
            ],
            "tasks": [
                "Solve 10 reasoning questions daily",
                "Practice timed reasoning sets",
                "Review incorrect answers"
            ]
        },

        "aptitude": {
            "title": "Quantitative Aptitude",
            "goal": "Improve numerical accuracy and calculation speed.",
            "topics": [
                "Percentages",
                "Profit and loss",
                "Ratio and proportion",
                "Time and work",
                "Probability"
            ],
            "tasks": [
                "Solve 15 aptitude questions",
                "Practice one timed test",
                "Review calculation mistakes"
            ]
        },

        "programming": {
            "title": "Programming",
            "goal": "Strengthen programming fundamentals and coding ability.",
            "topics": [
                "Variables and data types",
                "Loops",
                "Functions",
                "Arrays",
                "Strings",
                "Basic algorithms"
            ],
            "tasks": [
                "Solve 5 coding problems",
                "Practice arrays and strings",
                "Implement one small program"
            ]
        },

        "dbms": {
            "title": "Database Management",
            "goal": "Build stronger database fundamentals.",
            "topics": [
                "SQL",
                "Keys",
                "Normalization",
                "Joins",
                "Transactions",
                "Indexing"
            ],
            "tasks": [
                "Write 10 SQL queries",
                "Practice JOIN operations",
                "Review normalization"
            ]
        },

        "communication": {
            "title": "Communication Skills",
            "goal": "Improve professional communication.",
            "topics": [
                "Grammar",
                "Vocabulary",
                "Professional speaking",
                "Presentation",
                "Interview communication"
            ],
            "tasks": [
                "Practice speaking for 10 minutes",
                "Learn 10 new professional words",
                "Record one mock interview answer"
            ]
        },

        "leadership": {
            "title": "Leadership",
            "goal": "Develop leadership and teamwork skills.",
            "topics": [
                "Decision making",
                "Team management",
                "Conflict resolution",
                "Delegation",
                "Problem solving"
            ],
            "tasks": [
                "Study one leadership case",
                "Practice situational questions",
                "Write a short team-management scenario"
            ]
        },

        "problem_solving": {
            "title": "Problem Solving",
            "goal": "Improve structured problem-solving ability.",
            "topics": [
                "Problem identification",
                "Root cause analysis",
                "Decision making",
                "Critical thinking",
                "Case analysis"
            ],
            "tasks": [
                "Solve 3 case-based problems",
                "Practice root-cause analysis",
                "Review incorrect solutions"
            ]
        },

        "marketing": {
            "title": "Marketing",
            "goal": "Build fundamental marketing knowledge.",
            "topics": [
                "Marketing fundamentals",
                "Consumer behavior",
                "Digital marketing",
                "Branding",
                "Market research"
            ],
            "tasks": [
                "Study one marketing case",
                "Review marketing concepts",
                "Create one simple marketing plan"
            ]
        }
    }

    # ---------------------------------------------------------
    # Generate up to 4 weeks
    # ---------------------------------------------------------

    for index, (skill, score) in enumerate(
        focus_skills[:4],
        start=1
    ):

        skill_key = str(skill).lower().strip()

        content = learning_content.get(
            skill_key,
            {
                "title": skill.replace("_", " ").title(),
                "goal": "Strengthen this skill through structured practice.",
                "topics": [
                    "Fundamentals",
                    "Practice",
                    "Application"
                ],
                "tasks": [
                    "Review the fundamentals",
                    "Complete practice questions",
                    "Take a timed assessment"
                ]
            }
        )

        roadmap.append(
            {
                "week": index,
                "title": content["title"],
                "goal": content["goal"],
                "score": score,
                "topics": content["topics"],
                "tasks": content["tasks"]
            }
        )

    return roadmap