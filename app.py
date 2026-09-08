import gradio as gr

from utils.detection import detect_image
from utils.safety_rules import analyze_safety
from utils.qa import answer_question


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image, confidence):

    if image is None:
        return (
            None,
            "⚠️ Please upload an image.",
            "",
            [],
            {},
            []
        )

    # --------------------------------------------------------
    # YOLO11s Detection
    # --------------------------------------------------------

    annotated_image, detections = detect_image(
        image,
        confidence
    )

    # --------------------------------------------------------
    # Safety Rule Analysis
    # --------------------------------------------------------

    safety = analyze_safety(detections)

    # --------------------------------------------------------
    # Detection Report
    # --------------------------------------------------------

    detection_report = ""

    if detections:

        for detection in detections:

            class_name = detection["class_name"]
            conf = detection["confidence"]

            detection_report += (
                f"- **{class_name}** "
                f"({conf:.1%})\n"
            )

    else:

        detection_report = "No objects detected."

    # --------------------------------------------------------
    # Safety Report
    # --------------------------------------------------------

    safety_report = safety.get(
        "status",
        "Safety analysis completed."
    )

    violations = safety.get(
        "violations",
        []
    )

    if violations:

        safety_report += "\n\n### ⚠️ Violations\n\n"

        for violation in violations:

            safety_report += (
                f"- {violation}\n"
            )

    else:

        safety_report += (
            "\n\n### ✅ Safety Status\n\n"
            "No PPE violations detected."
        )

    # --------------------------------------------------------
    # Reset Chat
    # --------------------------------------------------------

    chat_history = []

    return (
        annotated_image,
        detection_report,
        safety_report,
        detections,
        safety,
        chat_history
    )


# ============================================================
# SAFE SIGHT Q&A
# ============================================================

def ask_safesight(
    question,
    history,
    detections,
    safety
):

    # --------------------------------------------------------
    # Initialize history
    # --------------------------------------------------------

    if history is None:
        history = []

    # --------------------------------------------------------
    # Empty question
    # --------------------------------------------------------

    if not question or not question.strip():

        return (
            history,
            ""
        )

    # --------------------------------------------------------
    # Check if image has been analyzed
    # --------------------------------------------------------

    if not detections:

        answer = (
            "⚠️ **Please upload and analyze an image first.**\n\n"
            "I need the detected objects and safety analysis "
            "before I can answer questions about the site."
        )

    else:

        # ----------------------------------------------------
        # LLM / QA
        # ----------------------------------------------------

        try:

            answer = answer_question(
                question,
                detections,
                safety
            )

        except Exception as e:

            answer = (
                "❌ **Unable to generate an answer.**\n\n"
                f"Error: `{str(e)}`"
            )

    # --------------------------------------------------------
    # GRADIO MESSAGES FORMAT
    #
    # Your installed Gradio expects:
    #
    # {
    #     "role": "user",
    #     "content": "..."
    # }
    #
    # NOT:
    #
    # ("question", "answer")
    # --------------------------------------------------------

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return (
        history,
        ""
    )


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():
    return []


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="SafeSight"
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
        # 🦺 SafeSight

        ### AI-Powered Construction Site Safety Monitoring

        Upload a construction-site image to detect workers,
        PPE, machinery and potential safety violations.
        """
    )

    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    gr.Markdown(
        """
        ## 📷 Construction Site Analysis
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # IMAGE INPUT
        # ----------------------------------------------------

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Construction Image"
            )

            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            detect_button = gr.Button(
                "🔍 Analyze Image",
                variant="primary"
            )

        # ----------------------------------------------------
        # DETECTION RESULT
        # ----------------------------------------------------

        with gr.Column():

            output_image = gr.Image(
                label="Detection Result"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "### 🔎 Detected Objects"
            )

            detection_output = gr.Markdown()

        with gr.Column():

            gr.Markdown(
                "### 🛡️ Safety Analysis"
            )

            safety_output = gr.Markdown()

    # ========================================================
    # APPLICATION STATE
    # ========================================================

    detections_state = gr.State([])

    safety_state = gr.State({})

    # ========================================================
    # Q&A
    # ========================================================

    gr.Markdown(
        """
        ---

        ## 💬 Ask SafeSight

        Ask questions about the objects and safety violations
        detected in the uploaded image.

        **Example questions:**

        - How many people are detected?
        - Are all workers wearing hardhats?
        - Are there any safety violations?
        - How many workers are without masks?
        - Is the site following PPE requirements?
        - What safety issues should be addressed?
        - Is this construction site safe?
        - Which PPE violation is the most serious?
        """
    )

    # --------------------------------------------------------
    # CHATBOT
    #
    # IMPORTANT:
    # Do NOT add type="messages"
    #
    # Your installed Gradio version already expects
    # the messages format by default.
    # --------------------------------------------------------

    chatbot = gr.Chatbot(
        label="SafeSight Assistant",
        height=400
    )

    # ========================================================
    # QUESTION INPUT
    # ========================================================

    with gr.Row():

        question_input = gr.Textbox(
            placeholder=(
                "Example: Are there any workers "
                "without safety equipment?"
            ),
            label="Ask a question",
            scale=5
        )

        ask_button = gr.Button(
            "🤖 Ask",
            variant="primary",
            scale=1
        )

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    clear_button = gr.Button(
        "🗑️ Clear Chat"
    )

    # ========================================================
    # EVENTS
    # ========================================================

    # --------------------------------------------------------
    # ANALYZE IMAGE
    # --------------------------------------------------------

    detect_button.click(
        fn=process_image,

        inputs=[
            image_input,
            confidence
        ],

        outputs=[
            output_image,
            detection_output,
            safety_output,
            detections_state,
            safety_state,
            chatbot
        ]
    )

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    ask_button.click(
        fn=ask_safesight,

        inputs=[
            question_input,
            chatbot,
            detections_state,
            safety_state
        ],

        outputs=[
            chatbot,
            question_input
        ]
    )

    # --------------------------------------------------------
    # ENTER KEY
    # --------------------------------------------------------

    question_input.submit(
        fn=ask_safesight,

        inputs=[
            question_input,
            chatbot,
            detections_state,
            safety_state
        ],

        outputs=[
            chatbot,
            question_input
        ]
    )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    clear_button.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch()