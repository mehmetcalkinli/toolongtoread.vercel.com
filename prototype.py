from flask import Flask, render_template, request
from toolongtoreadV4 import extract_bullet_points, sent_tokenize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        bullet_points = extract_bullet_points(text, num_points=3)
        sentences = sent_tokenize(text)
        underlined_text = ""
        for sentence in sentences:
            if sentence in bullet_points:
                underlined_text += f"<u>- {sentence}</u><br>"
            else:
                underlined_text += f"{sentence}<br>"
        return render_template('index.html', underlined_text=underlined_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)