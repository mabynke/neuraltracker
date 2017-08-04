from flask import Flask, render_template, url_for, jsonify, request
import fakeNeuralNetwork

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
	return render_template('hello.html',name=name)

@app.route('/_AI_prediction')
def _AI_prediction():
	a = request.args.get('a', 0, type=str)
	return jsonify(result=fakeNeuralNetwork.AImagic(a))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__": # we only start this webserver when this is run directly (aka this part will not be in python AI code)
	app.run(debug = True)


'''
@app.route('/user/<username>')
def profile(username):
	return 'hello %s' % username

@app.route('/post/<int:my_id>')
def show_post(my_id):
	return '<h2> This is your id: %s <h2>' % my_id

@app.route('/method', method = ['GET','POST'])
def index():
	return 'Method used: %s' % request.method
'''