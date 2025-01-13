import json
import time
from flask import Flask, jsonify, render_template, request, Response, stream_with_context

app = Flask(__name__, template_folder="templates")


# -----------------前端頁面-----------
# 主頁


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/visualize-camera_view")
def visualize3():
    return render_template("camera_view.html")

# -----------------後端頁面-----------


class camera_view:

    def __init__(self):
        self.a_camera_frame = {"image": "NC", "timestamp": ""}

    def update_camera_frame(self, frame):
        self.a_camera_frame = frame

    def get_camera_frame(self):
        return self.a_camera_frame


a_camera_view = camera_view()


@app.post("/post_camera_frame")
def receive_camera_frame():
    try:
        content = request.get_json()
        a_camera_view.update_camera_frame(content)
    except Exception as e:
        print(f"Error receiving data: {str(e)}")
        return jsonify({"success": False, "error": str(e)})
    return "camera_frame updated successfully"


@app.route('/get_camera_stream')
def make_stream():
    # 與相機網頁前端建立event-stream
    @stream_with_context
    def generate():
        try:
            while True:
                yield "data:" + json.dumps(a_camera_view.get_camera_frame()) + "\n\n"
                time.sleep(0.1)
        except GeneratorExit:
            print('closed')

    # 用stream發給前端
    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
