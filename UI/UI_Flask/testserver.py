from flask import Flask, render_template, jsonify, request, url_for

app = Flask(__name__)

@app.route('/')
def render():
	return "render_template('neuralTrackerUI_bilder.html')"

if __name__=='__main__':
    app.run(debug=True)


'''
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(SITE_ROOT, 'static', 'data.json')
data = json.load(open(json_url))
'''
