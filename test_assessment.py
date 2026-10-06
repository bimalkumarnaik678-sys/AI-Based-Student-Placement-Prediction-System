from assessment.engine import AssessmentEngine


engine = AssessmentEngine()

print(
    "TOTAL QUESTIONS:",
    len(engine.questions)
)


questions = engine.generate_test(
    discipline="MCA",
    career_interest="Software / IT",
    skill="programming",
    number_of_questions=3
)

print("\nGENERATED QUESTIONS:\n")

for question in questions:

    print(question["id"])
    print(question["question"])
    print(question["options"])
    print()