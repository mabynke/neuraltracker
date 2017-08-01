from flask import Flask, render_template, url_for, jsonify, request
import heavyMaths

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
	return render_template('hello.html',name=name)

@app.route('/_add_numbers') #Add two numbers server side, ridiculous but well...
def add_numbers():
	a = request.args.get('a', 0, type=list)
	b = request.args.get('b', 0, type=int)
	return jsonify(result=heavyMaths.listMultiply(a,b))

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