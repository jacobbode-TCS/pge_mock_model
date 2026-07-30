from flask import Flask, jsonify, render_template, request

from graphs.main_graph import run_workflow

app = Flask(__name__, template_folder="templates")


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
            image_path=image_file if image_file and image_file.filename else None,
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "request": request_text,
        "next_agent": result.get("next_agent"),
        "analysis": result.get("analysis"),
        "search": result.get("search"),
        "estimate": result.get("estimate"),
        "construction": result.get("construction"),
        "calibration": result.get("calibration"),
        "message": result.get("message"),
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )