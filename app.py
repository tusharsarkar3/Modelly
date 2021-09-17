from flask import Flask, render_template,request
import csv
import pandas as pd

app = Flask(__name__)


@app.route('/')
def uploady_file():
    print("Hello")
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #f = pd.read_csv(request.files['csvfile'])
        #print(f)
        # data = []
        # with open(f) as file:
        #     csvfile = csv.reader(file)
        #     for row in csvfile:
        #         data.append(row)
        # print(data)
        #get number of layers, preferably produce a list

        layers=[i for i in range(1,10)]
        return render_template('layers.html', layers=layers)
        #return 'file uploaded successfully'
@app.route('/layers', methods=['GET', 'POST'])
def getlayers():
    if request.method == 'POST':
        
        return render_template('', layers=layers) #3 to test
    

if __name__ == '__main__':
    app.run(debug=True)