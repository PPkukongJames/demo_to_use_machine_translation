
from flask import Flask,request
from MachineTranslation import Machine_Translation

app = Flask(__name__)
engine = Machine_Translation('pythai')
@app.route('/translate')
def translate():
    return engine.engine(request.args.get('text'))

# main driver function
if __name__ == '__main__':
    app.run()