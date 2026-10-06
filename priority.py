def get_priority(score):

    if score < 40:
        return "Very High"

    elif score < 60:
        return "High"

    elif score < 80:
        return "Medium"

    return "Low"


def get_status(score):

    if score >= 80:
        return "Strong"

    elif score >= 60:
        return "Moderate"

    elif score >= 40:
        return "Needs Improvement"

    return "Critical Improvement Needed"