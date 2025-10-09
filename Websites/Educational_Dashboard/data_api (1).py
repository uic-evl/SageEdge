from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/api/hello')
def hello():
    return jsonify({"message": "Hello from Windows!", "status": "success"})

@app.route('/api/data')
def get_data():
    try:
        conn = sqlite3.connect('fake_direction_evl.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM directional_data")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        result = [dict(zip(col_names, row)) for row in rows]
        conn.close()
        return jsonify({"data": result, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)