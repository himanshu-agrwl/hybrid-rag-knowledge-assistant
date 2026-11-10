def is_abstention(answer: str) -> bool:
    """Check whether the model declined to answer."""
    abstention_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "the provided context does not",
        "the context does not contain",
        "i cannot answer",
        "i can't answer",
        "not enough information",
    ]

    answer = answer.lower()
    return any(phrase in answer for phrase in abstention_phrases)


def evaluate_generation(results: list[dict]) -> dict:
    """Evaluate expected answer/abstention behaviour."""

    if not results:
        raise ValueError("No generation results provided.")

    correct = 0
    answered = 0
    abstained = 0

    for item in results:
        expected = item["expected_behavior"]
        answer = item["answer"]

        predicted = "abstain" if is_abstention(answer) else "answer"

        if predicted == expected:
            correct += 1

        if predicted == "answer":
            answered += 1
        else:
            abstained += 1

    total = len(results)

    return {
        "total": total,
        "behavior_accuracy": correct / total,
        "answer_rate": answered / total,
        "abstention_rate": abstained / total,
    }