def analyze_safety(detections):
    """
    Analyze YOLO detections and generate a basic safety summary.

    detections: list of dictionaries containing:
        class_name
        confidence
    """

    counts = {}

    for detection in detections:
        name = detection["class_name"]
        counts[name] = counts.get(name, 0) + 1

    violations = []

    if counts.get("NO-Hardhat", 0) > 0:
        violations.append(
            f"{counts['NO-Hardhat']} NO-Hardhat detection(s)"
        )

    if counts.get("NO-Mask", 0) > 0:
        violations.append(
            f"{counts['NO-Mask']} NO-Mask detection(s)"
        )

    if counts.get("NO-Safety Vest", 0) > 0:
        violations.append(
            f"{counts['NO-Safety Vest']} NO-Safety Vest detection(s)"
        )

    if violations:
        status = "⚠️ Safety violations detected"
    else:
        status = "✅ No obvious PPE violations detected"

    return {
        "status": status,
        "counts": counts,
        "violations": violations
    }