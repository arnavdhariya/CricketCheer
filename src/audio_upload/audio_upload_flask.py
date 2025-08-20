from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def upload():
    if request.files:
        if request.files['file'].filename.endswith('.wav'):
            file = request.files["file"]
            file.save(f"/data/user_data/{file.filename}")
            return "200 uploaded successfully"
        else:
            return "400 upload failed, wrong file type"
    else:
        return "400 upload failed, no file"
