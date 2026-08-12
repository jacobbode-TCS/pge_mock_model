from flask import Flask, jsonify, render_template, request

from graphs.main_graph import run_workflow

app = Flask(__name__, template_folder="templates")

# VERBOSE = True

def response_to_markdown(response):
    """Convert workflow response payloads into Markdown-friendly text."""
    if response is None:
        return ""

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        lines = []
        for key, value in response.items():
            if isinstance(value, (dict, list)):
                value_text = str(value)
            else:
                value_text = str(value)
            lines.append(f"- **{key}**: {value_text}")
        return "\n".join(lines)

    return str(response)


@app.get("/")
def home():
    """Render the main page with a simple form for interacting with the workflow."""
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/workflow")
def workflow():

    request_text = (request.form.get("request") or "").strip()

    if not request_text:
        return jsonify({"error": "request is required"}), 400

    image_file = request.files.get("image")

    try:
        result = run_workflow(
            request=request_text,
            image_path=image_file if image_file and image_file.filename else None, #type: ignore
        )

    except BaseException as exc:
        print(exc)
        return jsonify({"error": str(exc)}), 500

    response_text = response_to_markdown(result.get("response"))

    return jsonify({
        "request": request_text,
        "chosen_agent": result.get("chosen_agent"),
        "response": response_text,
        "sources": result.get("sources")
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )