from graphs.main_graph import run_workflow
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Render home page template, which includes a form for user input
    return render_template('home.html')
    pass

@app.route('/construction', methods=['GET'])
def get_construction():
    pass

@app.route('/estimation', methods=['GET'])
def get_estimation():
    pass

@app.route('/image_classification', methods=['GET'])
def get_image_classification():
    pass

@app.route('/knowledge_search', methods=['GET'])
def get_knowledge_search():
    pass

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
