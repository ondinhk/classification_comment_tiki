from flask import Flask, render_template, request
import processing_input

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html", result=[""","""])


@app.route('/', methods=['POST'])
def predictandoutput():
    input_text = request.form["text"]
    # Gọi model lên dự đoán
    label = processing_input.pre_comment(input_text)

    return render_template("index.html", result=[input_text, label])


if __name__ == '__main__':
    app.run(debug=True)
