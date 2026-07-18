let streaming = false;

function toggleStream() {
  const btn = document.getElementById("streamBtn");
  const feed = document.getElementById("liveFeed");
  const placeholder = document.getElementById("cameraPlaceholder");
  const status = document.getElementById("statusText");

  if (!streaming) {
    fetch("/start_stream");
    feed.src = "/video_feed?" + new Date().getTime();
    feed.style.display = "block";
    placeholder.style.display = "none";
    btn.innerText = "⏹ Stop Live Stream";
    status.innerText = "Live";
    streaming = true;
  } else {
    fetch("/stop_stream");
    feed.src = "";
    feed.style.display = "none";
    placeholder.style.display = "flex";
    btn.innerText = "▶ Start Live Stream";
    status.innerText = "Stopped";
    streaming = false;
  }
}

/* DOWNLOAD CSV */
function downloadAttendance() {
  window.location.href = "/download_attendance";
}
