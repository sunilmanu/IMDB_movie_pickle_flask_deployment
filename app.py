from flask import Flask, render_template, request
import pickle
import sqlite3
import os
import numpy as np


# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'model.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and 'photo' in request.files:
        name = request.form['moviereview']
        filename = photos.save(request.files['photo'])
        image_features, labels = extract_features(filename, 1)
        img_features = np.reshape(image_features,(1, 4 * 4 * 512))
        out=model.predict_classes(img_features)
        if out == ([[1]]):
            y="Dog"
        if out ==([[0]]):
            y="Cat"
        return render_template('results.html',
		                        content=name,
                                prediction=y)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)