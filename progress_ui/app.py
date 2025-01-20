from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Shared text variable
text_content = {"text": "Initial Text"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify(text_content)

@app.route('/update_text', methods=['POST'])
def update_text():
    global text_content
    new_text = request.json.get("text", "")
    text_content["text"] = new_text
    return jsonify({"message": "Text updated successfully!"})

if __name__ == '__main__':
    app.run(port=7696, debug=True)
