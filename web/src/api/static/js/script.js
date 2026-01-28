const lecturerId = "{{ user.user_id }}";

window.onload = function () {
  loadCourses();
};

// ==============================
// GLOBAL STATE
// ==============================
let lastLoadedCourseId = null;
let currentCourseId = null;
let currentCourseName = "";
let isFetchingAttendance = false;

let lastTabSwitchTime = 0;
const debounceDelay = 300;
let isProgrammaticTabSwitch = false;

// ==============================
// REALTIME STATE (NEW ARCH)
// ==============================
let realtimeState = {
  running: false,
  pollTimer: null,

  lastToastName: "",
  lastToastAt: 0,
  toastCooldownMs: 1800,
};

// ==============================
// TAB SWITCH
// ==============================
function showTab(tabId, courseId = null, courseName = "") {
  const now = Date.now();
  if (now - lastTabSwitchTime < debounceDelay) {
    console.log("Debouncing showTab call for tabId:", tabId);
    return;
  }
  lastTabSwitchTime = now;

  // 1) STOP REALTIME nếu rời attendance
  if (tabId !== "attendance" && realtimeState.running) {
    console.log("Leaving attendance tab → stop realtime");
    closeAttendanceModal();
  }

  // Flag để tránh xử lý double khi bootstrap trigger
  isProgrammaticTabSwitch = true;

  const tabButton = document.querySelector(`#tab-${tabId}`);
  if (!tabButton) {
    console.error(`Không tìm thấy nút tab với id: tab-${tabId}`);
    return;
  }

  const tab = new bootstrap.Tab(tabButton);
  tab.show();

  setTimeout(() => {
    isProgrammaticTabSwitch = false;
  }, 0);

  // 2) TAB ATTENDANCE
  if (tabId === "attendance") {
    const isNewCourse = courseId !== null && courseId !== currentCourseId;

    if (courseId) {
      currentCourseId = courseId;
      currentCourseName = courseName;
    }

    const list = document.getElementById("attendance-list");
    const title = document.getElementById("attendance-title");

    if (!currentCourseId) {
      if (title) title.textContent = " Lịch sử điểm danh";
      if (list) {
        list.innerHTML =
          "<p>Vui lòng chọn một khóa học để xem lịch sử điểm danh.</p>";
      }
      return;
    }

    // 2.1 Đổi course → reset realtime UI
    if (isNewCourse) {
      resetRealtimeUI();
    }

    // 2.2 Attendance history
    if (title) title.textContent = ` Lịch sử điểm danh - ${currentCourseName}`;

    if (lastLoadedCourseId === currentCourseId) {
      console.log("Attendance already loaded for:", currentCourseId);
      return;
    }

    if (isFetchingAttendance) {
      console.log("Đang fetch attendance, bỏ qua...");
      return;
    }

    loadAttendance(currentCourseId, currentCourseName);
  }
}

// ==============================
// ATTENDANCE HISTORY
// ==============================
function loadAttendance(courseId, courseName) {
  if (isFetchingAttendance) {
    console.log("Đang fetch attendance, bỏ qua...");
    return;
  }

  isFetchingAttendance = true;
  console.log("Loading attendance for courseId:", courseId);

  fetch(`/api/courses/${courseId}/attendance`)
    .then((res) => {
      console.log("Fetch response status:", res.status);
      if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
      return res.json();
    })
    .then((data) => {
      console.log("Attendance data:", data);
      lastLoadedCourseId = courseId;

      const list = document.getElementById("attendance-list");
      const title = document.getElementById("attendance-title");
      if (!list || !title) {
        console.error("attendance-list or attendance-title not found");
        return;
      }

      title.textContent = ` Lịch sử điểm danh - ${courseName}`;
      list.innerHTML = "";

      if (!Array.isArray(data) || data.length === 0) {
        list.innerHTML = "<p>Không có dữ liệu điểm danh.</p>";
        return;
      }

      data.forEach((att) => {
        const div = document.createElement("div");
        div.className = "list-group-item";
        div.innerHTML = `
            <strong>${att.student_name}</strong> |
            <span>${new Date(att.time).toLocaleString("vi-VN", {
              timeZone: "Asia/Ho_Chi_Minh",
            })}</span> |
            ${
              att.recognized
                ? "<span class='text-success'> Đã nhận diện</span>"
                : "<span class='text-danger'> Chưa nhận</span>"
            }
            ${
              att.image_base64
                ? `<br><img src="${att.image_base64}" class="rounded mt-2" style="max-height:100px;">`
                : ""
            }
          `;
        list.appendChild(div);
      });
    })
    .catch((err) => {
      console.error("Error loading attendance:", err);
      const list = document.getElementById("attendance-list");
      if (list) {
        list.innerHTML = `<p class="text-danger">Lỗi tải dữ liệu điểm danh: ${err}</p>`;
      }
    })
    .finally(() => {
      isFetchingAttendance = false;
    });
}

// ==============================
// COURSES
// ==============================
function loadCourses() {
  fetch(`/api/lecturer/${lecturerId}/courses`)
    .then((res) => res.json())
    .then((data) => {
      const courseList = document.getElementById("course-list");
      if (!courseList) return;

      courseList.innerHTML = "";
      (data || []).forEach((course) => {
        const col = document.createElement("div");
        col.className = "col-md-6 mb-3";
        col.innerHTML = `
          <div class="card shadow-sm">
            <div class="card-header">
              <h5 class="card-title mb-0">${course.name}</h5>
            </div>
            <div class="card-body">
              <p><strong>Học kỳ:</strong> ${course.semester}</p>
              <button class="btn btn-outline-primary btn-sm" onclick="loadStudents(${course.id}, '${escapeQuotes(
          course.name
        )}')"> Xem SV</button>

              <button class="btn btn-outline-secondary btn-sm ms-2"
                onclick="showTab('attendance', ${course.id}, '${escapeQuotes(
          course.name
        )}')">
                 Lịch sử điểm danh
              </button>
            </div>
          </div>
        `;
        courseList.appendChild(col);
      });
    });
}

function escapeQuotes(str) {
  return String(str || "").replace(/'/g, "\\'");
}

// ==============================
// STUDENTS + REGISTER FACE
// ==============================
function loadStudents(courseId, courseName) {
  currentCourseId = courseId;
  currentCourseName = courseName;

  fetch(`/api/courses/${courseId}/students_with_embedding`)
    .then((res) => res.json())
    .then((data) => {
      showTab("students");

      const title = document.getElementById("student-title");
      if (title)
        title.textContent = ` Danh sách sinh viên - ${courseName}`;

      const studentList = document.getElementById("student-list");
      if (!studentList) return;

      studentList.innerHTML = "";

      (data || []).forEach((st) => {
        const div = document.createElement("div");
        div.className =
          "list-group-item d-flex justify-content-between align-items-center";

        const info = document.createElement("span");
        info.innerHTML = `<strong>${st.id}</strong> - ${st.full_name}`;

        const btn = document.createElement("button");
        btn.className = `btn btn-sm ${
          st.has_embedding ? "btn-warning" : "btn-primary"
        }`;
        btn.innerHTML = st.has_embedding ? " Đăng ký lại" : " Đăng ký";
        btn.onclick = () =>
          handleRegisterFace(st.id, st.full_name, st.has_embedding);

        div.appendChild(info);
        div.appendChild(btn);
        studentList.appendChild(div);
      });
    });
}

function showQR(courseId, courseName) {
  const qrContainer = document.getElementById("qr-container");
  if (!qrContainer) return;
  qrContainer.innerHTML = "";

  const img = document.createElement("img");
  img.src = `/api/qr_ip?course_id=${encodeURIComponent(
    courseId
  )}&course_name=${encodeURIComponent(courseName)}`;
  img.alt = "QR chứa thông tin điểm danh";
  img.style.width = "256px";
  img.style.height = "256px";

  img.onload = () => {
    qrContainer.appendChild(img);
    const qrModal = new bootstrap.Modal(document.getElementById("qrModal"));
    qrModal.show();
  };
  img.onerror = () => {
    qrContainer.innerHTML = "<p class='text-danger'>Lỗi: Không thể tải mã QR</p>";
    const qrModal = new bootstrap.Modal(document.getElementById("qrModal"));
    qrModal.show();
  };
}

let currentStudentId = null;
let currentStudentName = null;
let stream = null;

async function handleRegisterFace(studentId, fullName, hasEmbedding) {
  currentStudentId = studentId;
  currentStudentName = fullName;

  const modal = new bootstrap.Modal(document.getElementById("faceModal"));
  modal.show();

  const video = document.getElementById("video");
  try {
    stream = await setupWebcam(video);
  } catch (err) {
    console.error("Failed to setup webcam:", err);
    const status = document.getElementById("register-status");
    if (status) {
      status.innerHTML = "<span class='text-danger'> Không thể truy cập webcam</span>";
    }
  }
}

function captureAndSend() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const status = document.getElementById("register-status");
  const ctx = canvas.getContext("2d");

  const capturedImages = [];
  let count = 0;
  const totalCaptures = 10;
  const delay = 600;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (status) status.innerHTML = "<span class='text-info'> Đang chụp ảnh...</span>";

  const capture = () => {
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        capturedImages.push(blob);
        count++;

        if (count < totalCaptures) {
          setTimeout(capture, delay);
        } else {
          const formData = new FormData();
          formData.append("student_id", currentStudentId);
          formData.append("name", currentStudentName);
          for (let i = 0; i < capturedImages.length; i++) {
            formData.append("files", capturedImages[i], `frame_${i}.jpg`);
          }

          fetch("/register", {
            method: "POST",
            body: formData,
          })
            .then((res) => res.json())
            .then((result) => {
              if (!status) return;
              if (result.success) {
                status.innerHTML = "<span class='text-success'> Đăng ký thành công!</span>";
              } else {
                status.innerHTML =
                  "<span class='text-danger'> Đăng ký thất bại: " +
                  result.message +
                  "</span>";
              }
            })
            .catch((err) => {
              console.error(err);
              if (status)
                status.innerHTML =
                  "<span class='text-danger'> Lỗi kết nối đến máy chủ</span>";
            });
        }
      },
      "image/jpeg"
    );
  };

  capture();
}

document.getElementById("faceModal")?.addEventListener("hidden.bs.modal", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
});

async function setupWebcam(videoElement) {
  if (videoElement?.srcObject) return videoElement.srcObject;

  const s = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    },
  });

  videoElement.srcObject = s;

  await new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      resolve();
    };
  });

  return s;
}

// ==============================
// REALTIME ATTENDANCE (NEW)
// - NO webcam in browser
// - NO canvas/blob
// - ONLY start/stop + MJPEG + poll status
// ==============================
function resetRealtimeUI() {
  const list = document.getElementById("realtime-result");
  if (list) list.innerHTML = "";

  realtimeState.lastToastName = "";
  realtimeState.lastToastAt = 0;
}

async function openAttendanceModal() {
  if (!currentCourseId) {
    alert("Vui lòng chọn một khóa học trước khi điểm danh!");
    return;
  }

  // mở modal
  const modal = new bootstrap.Modal(document.getElementById("attendance-modal"));
  modal.show();

  // start engine
  await fetch("/api/realtime/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ course_id: currentCourseId }),
  });

  // show MJPEG stream
  const img = document.getElementById("attendance-video");
  if (img) {
    // cache-bust để tránh browser giữ stream cũ
    img.src = `/video_feed?t=${Date.now()}`;
  }

  // poll results
  startRealtimePolling();
}

async function closeAttendanceModal() {
  // stop poll
  stopRealtimePolling();

  // stop engine
  try {
    await fetch("/api/realtime/stop", { method: "POST" });
  } catch (e) {}

  // clear stream
  const img = document.getElementById("attendance-video");
  if (img) img.src = "";

  // hide modal
  const modalEl = document.getElementById("attendance-modal");
  const modal = bootstrap.Modal.getInstance(modalEl);
  if (modal) modal.hide();

  realtimeState.running = false;
}

function startRealtimePolling() {
  stopRealtimePolling();

  realtimeState.running = true;

  const tick = async () => {
    if (!realtimeState.running) return;

    try {
      const res = await fetch("/api/realtime/status");
      const data = await res.json();

      // expect: {running: bool, results: [...]}
      renderRealtimeResults(data?.results || []);

      // nếu backend stop thì tự stop
      if (data?.running === false) {
        realtimeState.running = false;
      }
    } catch (e) {
      console.warn("Realtime status error:", e);
    }

    realtimeState.pollTimer = setTimeout(tick, 500);
  };

  realtimeState.pollTimer = setTimeout(tick, 200);
}

function stopRealtimePolling() {
  realtimeState.running = false;
  if (realtimeState.pollTimer) {
    clearTimeout(realtimeState.pollTimer);
    realtimeState.pollTimer = null;
  }
}

function renderRealtimeResults(results) {
  const list = document.getElementById("realtime-result");
  if (!list) return;

  list.innerHTML = "";

  if (!Array.isArray(results) || results.length === 0) return;

  results.forEach((r) => {
    const name = r?.name || "Unknown";
    const trackId = r?.track_id ?? "";
    const sim = r?.similarity;

    const div = document.createElement("div");
    div.className = "list-group-item d-flex justify-content-between align-items-center";

    const left = document.createElement("div");
    left.innerHTML = `<strong>${name}</strong> <span class="text-muted">#${trackId}</span>`;

    const right = document.createElement("div");
    right.className = "text-muted";
    right.textContent =
      sim != null && name !== "Unknown" ? `sim=${Number(sim).toFixed(2)}` : "";

    div.appendChild(left);
    div.appendChild(right);
    list.appendChild(div);
  });

  maybeToast(results);
}

function maybeToast(results) {
  if (!Array.isArray(results) || results.length === 0) return;

  // ưu tiên face đầu tiên
  const r = results[0];
  const name = r?.name || "Unknown";
  if (!name || name === "Unknown") return;

  const now = Date.now();
  if (
    name === realtimeState.lastToastName &&
    now - realtimeState.lastToastAt < realtimeState.toastCooldownMs
  ) {
    return;
  }

  realtimeState.lastToastName = name;
  realtimeState.lastToastAt = now;

  try {
    showToast(`Đã điểm danh: ${name}`);
  } catch (e) {}
}
let rtState = {
  running: false,
  pollTimer: null,
  lastToastName: "",
  lastToastAt: 0,
  toastCooldownMs: 1800,
};

async function rtStart() {
  const courseId = Number(document.getElementById("rt-course-id")?.value || 1);

  await fetch("/api/realtime/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ course_id: courseId }),
  });

  // bật stream
  const img = document.getElementById("rt-stream");
  if (img) img.src = `/video_feed?t=${Date.now()}`;

  rtState.running = true;
  rtPoll();
}

async function rtStop() {
  rtState.running = false;
  if (rtState.pollTimer) {
    clearTimeout(rtState.pollTimer);
    rtState.pollTimer = null;
  }

  try {
    await fetch("/api/realtime/stop", { method: "POST" });
  } catch (e) {}

  const img = document.getElementById("rt-stream");
  if (img) img.src = "";

  const results = document.getElementById("rt-results");
  if (results) results.innerHTML = "";

  const st = document.getElementById("rt-status");
  if (st) st.textContent = "{}";
}

async function rtPoll() {
  if (!rtState.running) return;

  try {
    const res = await fetch("/api/realtime/status");
    const data = await res.json();

    // show raw status
    const st = document.getElementById("rt-status");
    if (st) st.textContent = JSON.stringify(data, null, 2);

    // render results
    renderRtResults(data?.results || []);

    // nếu backend đã stop
    if (data?.running === false) {
      rtState.running = false;
    }
  } catch (e) {
    console.warn("rtPoll error:", e);
  }

  rtState.pollTimer = setTimeout(rtPoll, 500);
}

function renderRtResults(results) {
  const list = document.getElementById("rt-results");
  if (!list) return;

  list.innerHTML = "";
  if (!Array.isArray(results) || results.length === 0) return;

  results.forEach((r) => {
    const name = r?.name || "Unknown";
    const trackId = r?.track_id ?? "";
    const sim = r?.similarity;

    const div = document.createElement("div");
    div.className = "list-group-item d-flex justify-content-between align-items-center";

    const left = document.createElement("div");
    left.innerHTML = `<strong>${name}</strong> <span class="text-muted">#${trackId}</span>`;

    const right = document.createElement("div");
    right.className = "text-muted";
    right.textContent =
      sim != null && name !== "Unknown" ? `sim=${Number(sim).toFixed(2)}` : "";

    div.appendChild(left);
    div.appendChild(right);
    list.appendChild(div);
  });

  // toast chống spam (nếu bạn có showToast)
  maybeRtToast(results);
}

function maybeRtToast(results) {
  if (!Array.isArray(results) || results.length === 0) return;

  const name = results[0]?.name || "Unknown";
  if (!name || name === "Unknown") return;

  const now = Date.now();
  if (name === rtState.lastToastName && (now - rtState.lastToastAt) < rtState.toastCooldownMs) return;

  rtState.lastToastName = name;
  rtState.lastToastAt = now;

  try { showToast(`Đã điểm danh: ${name}`); } catch (e) {}
}