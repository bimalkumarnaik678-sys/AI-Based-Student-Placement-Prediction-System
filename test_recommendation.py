from recommendation.engine import generate_recommendations
from recommendation.roadmap import generate_roadmap


assessment_scores = {
    "programming": 42,
    "aptitude": 86,
    "reasoning": 61,
    "communication": 72,
    "leadership": 55
}


recommendations = generate_recommendations(
    assessment_scores
)


print("PERSONALIZED RECOMMENDATIONS")
print("============================")

for item in recommendations:

    print("\nSkill:", item["title"])
    print("Score:", item["score"])
    print("Priority:", item["priority"])
    print("Status:", item["status"])

    print("Actions:")

    for action in item["actions"]:
        print("-", action)


roadmap = generate_roadmap(
    assessment_scores
)


print("\n\n30-DAY ROADMAP")
print("==============")

for week, tasks in roadmap.items():

    print("\n", week)

    for task in tasks:
        print("-", task)