class AssessmentScorer:

    @staticmethod
    def calculate_score(questions, answers):

        total = len(questions)

        correct = 0
        wrong = 0
        unanswered = 0

        details = []

        for question in questions:

            question_id = str(question["id"])

            # -------------------------------------------------
            # Get the submitted answer using QUESTION ID
            # -------------------------------------------------

            raw_answer = answers.get(question_id)

            selected_index = None

            if raw_answer is not None:

                try:
                    selected_index = int(raw_answer)
                except (ValueError, TypeError):
                    selected_index = None

            # -------------------------------------------------
            # JSON answer is ALWAYS 0-based
            #
            # 0 = first option
            # 1 = second option
            # 2 = third option
            # 3 = fourth option
            # -------------------------------------------------

            try:
                correct_index = int(question["answer"])
            except (ValueError, TypeError):
                correct_index = None

            options = question.get("options", [])

            # -------------------------------------------------
            # Selected option text
            # -------------------------------------------------

            selected_text = None

            if (
                selected_index is not None
                and 0 <= selected_index < len(options)
            ):
                selected_text = options[selected_index]

            # -------------------------------------------------
            # Correct option text
            # -------------------------------------------------

            correct_text = ""

            if (
                correct_index is not None
                and 0 <= correct_index < len(options)
            ):
                correct_text = options[correct_index]

            # -------------------------------------------------
            # Determine result
            # -------------------------------------------------

            if selected_index is None:

                unanswered += 1

                is_correct = False

            elif correct_index is not None and selected_index == correct_index:

                correct += 1

                is_correct = True

            else:

                wrong += 1

                is_correct = False

            # -------------------------------------------------
            # Add question review
            # -------------------------------------------------

            details.append({
                "id": question_id,
                "question": question.get("question", ""),
                "is_correct": is_correct,
                "selected_text": selected_text,
                "correct_text": correct_text,
                "explanation": question.get("explanation", "")
            })

        # -----------------------------------------------------
        # Percentage
        # -----------------------------------------------------

        percentage = (
            round((correct / total) * 100, 2)
            if total > 0
            else 0
        )

        return {
            "score": correct,
            "correct": correct,
            "wrong": wrong,
            "unanswered": unanswered,
            "total": total,
            "percentage": percentage,
            "details": details
        }