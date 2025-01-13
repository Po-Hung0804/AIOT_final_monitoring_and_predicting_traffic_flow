let source = new EventSource("/get_camera_stream"); //前端向flask的/get_camera_stream發送請求
source.onmessage = function (event) {

    //取得元素
    let timeDiv = document.getElementById("timeDiv");
    let camera_canvas = document.getElementById("main_camera");
    var ctx = camera_canvas.getContext("2d");
    let frame_data = JSON.parse(event.data)

    if (frame_data["image"] == "NC") {
        timeDiv.innerText = "相機未連線"
    } else {
        let timestamp = frame_data["timestamp"]
        let frame_base64 = frame_data["image"]

        //更改數值
        var image = new Image();
        image.onload = function () {
            camera_canvas.width = image.naturalWidth
            camera_canvas.height = image.naturalHeight
            camera_canvas.style.aspectRatio = image.naturalWidth + "/" + image.naturalHeight
            ctx.drawImage(image, 0, 0);
            timeDiv.innerText = timestamp
        };
        image.src = "data:image/png;base64," + frame_base64
    }
    //source.close()
}