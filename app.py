from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    # test_np_input = np.array([[1], [2], [17]])
    # model = load('model.joblib')
    # preds = model.predict(test_np_input)
    # preds_as_str = str(preds)
    # return preds_as_str

    req = request.method
    if req == "GET":
         return render_template('index.html', href = "static/base_pic.svg")
    else:
        text = request.form['text']

        # if the output img is being cached then use the below code that randomly generates string 
        # and pass it to file_path to keep on getting new images loaded in the browser
        #random_string = uuid.uuid4().hex

        file_path = 'static/prediction_pic.svg'
        make_picture('AgesAndHeights.pkl', load('model.joblib'), floats_string_to_np_arr(text), file_path)
        return render_template('index.html', href = file_path)

#processing input text and returning a numpy array for the model
def floats_string_to_np_arr(floats_str):
  def isFloat(s):
    try:
      float(s)
      return True
    except:
      return False

  floats = np.array([float(x) for x in floats_str.split(',') if isFloat(x)])
  return floats.reshape(len(floats), 1)


# creates new output model markers by taking the processed new input values
def make_picture(training_data_filename, model, new_np_input_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']

  x_new = np.array(list(range(19))).reshape(19, 1)
  preds = model.predict(x_new)

  figure = px.scatter(x=ages, y=heights, title="Ages vs Heights, of people", labels={'x': 'Age (years)', 'y': 'Height (inches)'})

  figure.add_trace(go.Scatter(x = x_new.reshape(19), y = preds, mode = 'lines', name = 'Model'))

  new_preds = model.predict(new_np_input_arr)
  figure.add_trace(go.Scatter(x = new_np_input_arr.reshape(len(new_np_input_arr)), y = new_preds, mode= 'markers', name = 'New Output Model', 
                                                           marker=dict(color = 'purple', size = 15, line = dict(color = 'purple', width = 2))))
  

  figure.write_image(output_file, width = 800, engine = 'kaleido')
  figure.show()     

if __name__ == '__main__':
    app.run(debug=True)


