import pickle
from flask import Flask, render_template, request
app = Flask(__name__)

f = open('model.pkl','rb')
clf = pickle.load(f)
f.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        # print(myDict)

        f = float(myDict['fever'])
        a = int(myDict['age'])
        p = int(myDict['bodyPain'])
        rn = int(myDict['runnyNose'])
        db = int(myDict['diffBreath'])
        
        inputFeatures = [f,p,a,rn,db]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf = round(infProb*100))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)