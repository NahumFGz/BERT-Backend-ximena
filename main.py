from nlpmodule.tools.BertClassifier import BertClassifier
from services.api import app

def main():
    #app.run(port=3000, debug=True)
    app.run(host="0.0.0.0", port=3000, debug=True)

if __name__ == '__main__':
    main()