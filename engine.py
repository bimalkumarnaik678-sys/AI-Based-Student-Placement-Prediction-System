import json
import os
import random


class AssessmentEngine:

    QUESTIONS_PER_TEST = 5
    MIN_QUESTION_BANK = 50

    def __init__(self):
        self.base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.question_bank_dir = os.path.join(
            self.base_dir,
            "question_bank"
        )

        self.skill_files = {
            "aptitude": "aptitude.json",
            "reasoning": "reasoning.json",
            "communication": "communication.json",
            "leadership": "leadership.json",
            "problem_solving": "problem_solving.json",

            # Keep these if their JSON files already exist
            "programming": "programming.json",
            "dbms": "dbms.json",
            "marketing": "marketing.json",
        }

        self.questions = {}

        self.load_question_banks()


    # ================================================================
    # LOAD ALL QUESTION BANKS
    # ================================================================

    def load_question_banks(self):

        self.questions = {}

        for skill, filename in self.skill_files.items():

            path = os.path.join(
                self.question_bank_dir,
                filename
            )

            if not os.path.exists(path):

                self.questions[skill] = []

                print(
                    f"[WARNING] Question bank not found: {path}"
                )

                continue

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                if not isinstance(data, list):

                    print(
                        f"[WARNING] {filename} must contain a JSON list."
                    )

                    self.questions[skill] = []

                    continue

                self.questions[skill] = data

                print(
                    f"[ASSESSMENT] {skill}: "
                    f"{len(data)} questions loaded"
                )

            except json.JSONDecodeError as exc:

                print(
                    f"[ERROR] Invalid JSON in {filename}: {exc}"
                )

                self.questions[skill] = []

            except Exception as exc:

                print(
                    f"[ERROR] Could not load {filename}: {exc}"
                )

                self.questions[skill] = []


    # ================================================================
    # NORMALIZE ANSWER INDEX
    # ================================================================

    @staticmethod
    def normalize_question(question):

        question = dict(question)

        options = question.get(
            "options",
            []
        )

        answer = question.get(
            "answer"
        )

        if not isinstance(options, list):

            return None

        if not isinstance(answer, int):

            return None

        if answer < 0 or answer >= len(options):

            return None

        question["answer"] = answer

        question["options"] = options

        return question


    # ================================================================
    # FILTER QUESTIONS
    # ================================================================

    def filter_questions(
        self,
        skill,
        discipline="ALL",
        career_interest="Other"
    ):

        bank = self.questions.get(
            skill,
            []
        )

        valid_questions = []

        for question in bank:

            normalized = self.normalize_question(
                question
            )

            if normalized is None:
                continue

            # --------------------------------------------------------
            # Discipline filter
            # --------------------------------------------------------

            disciplines = normalized.get(
                "discipline",
                ["ALL"]
            )

            if not isinstance(
                disciplines,
                list
            ):

                disciplines = ["ALL"]

            discipline_match = (

                "ALL" in disciplines

                or discipline in disciplines

            )

            if not discipline_match:
                continue

            # --------------------------------------------------------
            # Career filter
            # --------------------------------------------------------

            careers = normalized.get(
                "career_tracks",
                ["ALL"]
            )

            if not isinstance(
                careers,
                list
            ):

                careers = ["ALL"]

            career_match = (

                "ALL" in careers

                or career_interest in careers

            )

            if not career_match:
                continue

            valid_questions.append(
                normalized
            )

        return valid_questions


    # ================================================================
    # GENERATE RANDOM TEST
    # ================================================================

    def generate_test(
        self,
        discipline="ALL",
        career_interest="Other",
        skill=None,
        number_of_questions=5
    ):

        if not skill:

            return []

        filtered_questions = self.filter_questions(
            skill=skill,
            discipline=discipline,
            career_interest=career_interest
        )

        if not filtered_questions:

            return []

        # ------------------------------------------------------------
        # Randomly select questions
        # ------------------------------------------------------------

        question_count = min(
            number_of_questions,
            len(filtered_questions)
        )

        selected_questions = random.sample(
            filtered_questions,
            question_count
        )

        # ------------------------------------------------------------
        # Randomize option order
        #
        # IMPORTANT:
        # When options are shuffled, the answer index must also change.
        # ------------------------------------------------------------

        final_questions = []

        for question in selected_questions:

            question = dict(question)

            original_options = list(
                question["options"]
            )

            original_answer = int(
                question["answer"]
            )

            correct_option = (
                original_options[
                    original_answer
                ]
            )

            option_pairs = list(
                enumerate(
                    original_options
                )
            )

            random.shuffle(
                option_pairs
            )

            shuffled_options = [
                option
                for _, option in option_pairs
            ]

            new_answer = next(
                index
                for index, (_, option) in enumerate(
                    option_pairs
                )
                if option == correct_option
            )

            question["options"] = shuffled_options

            question["answer"] = new_answer

            final_questions.append(
                question
            )

        return final_questions


    # ================================================================
    # QUESTION BANK INFORMATION
    # ================================================================

    def get_question_count(
        self,
        skill
    ):

        return len(
            self.questions.get(
                skill,
                []
            )
        )


    def reload(self):

        self.load_question_banks()