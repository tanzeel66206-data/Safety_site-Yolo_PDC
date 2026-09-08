import os
from collections import Counter

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.7-flash"


# ============================================================
# BUILD STRUCTURED CONTEXT
# ============================================================

def build_context(detections, safety):
    """
    Convert YOLO detections and safety-rule results
    into a compact context for the LLM.
    """

    counts = Counter()

    confidence_values = {}

    for detection in detections:

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        counts[class_name] += 1

        confidence_values.setdefault(
            class_name,
            []
        ).append(confidence)

    # --------------------------------------------------------
    # Detection Summary
    # --------------------------------------------------------

    detection_summary = []

    for class_name, count in counts.items():

        avg_confidence = sum(
            confidence_values[class_name]
        ) / len(confidence_values[class_name])

        detection_summary.append(
            f"- {class_name}: {count} "
            f"(average confidence: {avg_confidence:.1%})"
        )

    if not detection_summary:
        detection_summary.append(
            "- No objects detected"
        )

    # --------------------------------------------------------
    # Safety Status
    # --------------------------------------------------------

    safety_status = safety.get(
        "status",
        "No safety analysis available."
    )

    violations = safety.get(
        "violations",
        []
    )

    violation_summary = []

    for violation in violations:
        violation_summary.append(
            f"- {violation}"
        )

    if not violation_summary:
        violation_summary.append(
            "- No PPE violations detected"
        )

    # --------------------------------------------------------
    # Final Context
    # --------------------------------------------------------

    context = f"""
SAFESIGHT COMPUTER VISION ANALYSIS

Detected Objects:
{chr(10).join(detection_summary)}

Safety Status:
{safety_status}

Safety Violations:
{chr(10).join(violation_summary)}
"""

    return context


# ============================================================
# LLM QUESTION ANSWERING
# ============================================================

def answer_question(
    question,
    detections,
    safety,
    conversation_history=None
):
    """
    Answer a user's natural-language question using
    YOLO detection data and safety-rule analysis.
    """

    if not detections:

        return (
            "I don't have any analyzed detections yet. "
            "Please upload and analyze a construction-site "
            "image first."
        )

    context = build_context(
        detections,
        safety
    )

    # --------------------------------------------------------
    # Conversation Context
    # --------------------------------------------------------

    conversation = ""

    if conversation_history:

        recent_history = conversation_history[-6:]

        conversation_lines = []

        for message in recent_history:

            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            conversation_lines.append(
                f"{role.upper()}: {content}"
            )

        conversation = (
            "\nPrevious conversation:\n"
            + "\n".join(conversation_lines)
        )

    # --------------------------------------------------------
    # System Instruction
    # --------------------------------------------------------

    system_instruction = """
You are SafeSight Assistant, an AI assistant for
construction-site safety monitoring.

Your job is to answer questions using ONLY the
computer-vision information provided by SafeSight.

IMPORTANT RULES:

1. Do not invent objects, workers, PPE, or violations.
2. Treat YOLO detections as the source of truth.
3. Treat the safety-rule analysis as the source of truth
   for PPE violations.
4. You may reason over the provided information.
5. Understand different ways users can ask the same question.
6. Keep answers concise and easy to understand.
7. If the requested information is not available,
   clearly say that it cannot be determined.
8. Do not claim that you personally inspected the image.
9. Do not invent relationships between individual workers
   unless the detection data explicitly supports them.
10. For safety questions, prioritize accuracy over speculation.

Examples:

"How many workers are there?"
"How many people are present?"
"How many workers do you see?"

These can all be interpreted as asking for the
Person detection count.

Similarly:

"Is everyone wearing a mask?"
"Are the workers properly masked?"
"Do any workers lack masks?"

These are questions about Mask and NO-Mask detections.

Answer naturally and professionally.
"""

    # --------------------------------------------------------
    # User Prompt
    # --------------------------------------------------------

    prompt = f"""
{context}

{conversation}

USER QUESTION:
{question}

Answer the user's question using the SafeSight analysis.
"""

    # --------------------------------------------------------
    # Gemini Request
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.2,
                "max_output_tokens": 300,
            }
        )

        answer = response.text

        if not answer:
            return (
                "I couldn't generate an answer from "
                "the available detection data."
            )

        return answer.strip()

    except Exception as e:

        print(
            f"Gemini API Error: {e}"
        )

        return (
            "⚠️ I couldn't connect to the SafeSight "
            "reasoning model. Please check your Gemini "
            "API key and internet connection."
        )