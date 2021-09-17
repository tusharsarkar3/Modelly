from flask import Flask, render_template,request
import csv
import pandas as pd

app = Flask(__name__)


@app.route('/')
def uploady_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = pd.read_csv(request.files['csvfile'])
        print(f)
        # data = []
        # with open(f) as file:
        #     csvfile = csv.reader(file)
        #     for row in csvfile:
        #         data.append(row)
        # print(data)
        # return render_template('data.html', data=data)
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)