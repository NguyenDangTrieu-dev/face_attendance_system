const lecturerId = "{{ user.user_id }}";

window.onload = function () {
    loadCourses();
};
let lastLoadedCourseId = null;
let currentCourseId = null;
let currentCourseName = '';
let isFetchingAttendance = false;
let lastTabSwitchTime = 0;
const debounceDelay = 300;
let isProgrammaticTabSwitch = false; // New flag

function showTab(tabId, courseId = null, courseName = '') {
    const now = Date.now();
    if (now - lastTabSwitchTime < debounceDelay) {
        console.log("Debouncing showTab call for tabId:", tabId);
        return;
    }
    lastTabSwitchTime = now;

    // Set the flag to indicate this is a programmatic call
    isProgrammaticTabSwitch = true;

    const tabButton = document.querySelector(`#tab-${tabId}`);
    if (!tabButton) {
        console.error(`Không tìm thấy nút tab với id: tab-${tabId}`);
        return;
    }
    const tab = new bootstrap.Tab(tabButton);
    tab.show();

    // Reset the flag after the tab is shown
    setTimeout(() => {
        isProgrammaticTabSwitch = false;
    }, 0);

    if (tabId === 'attendance') {
        if (courseId) {
            currentCourseId = courseId;
            currentCourseName = courseName;
        }

        const list = document.getElementById("attendance-list");
        const title = document.getElementById("attendance-title");

        if (!currentCourseId) {
            title.textContent = " Lịch sử điểm danh";
            list.innerHTML = "<p>Vui lòng chọn một khóa học để xem lịch sử điểm danh.</p>";
            return;
        }

        if (lastLoadedCourseId === currentCourseId) {
            console.log(" Attendance đã được load rồi cho courseId:", currentCourseId);
            title.textContent = ` Lịch sử điểm danh - ${currentCourseName}`;
            return;
        }

        if (isFetchingAttendance) {
            console.log("Đang fetch attendance, bỏ qua...");
            return;
        }

        loadAttendance(currentCourseId, currentCourseName);
    }
}
function loadAttendance(courseId, courseName) {
    if (isFetchingAttendance) {
        console.log("Đang fetch attendance, bỏ qua...");
        return;
    }

    isFetchingAttendance = true;
    console.log("Loading attendance for courseId:", courseId);

    fetch(`/api/courses/${courseId}/attendance`)
        .then(res => {
            console.log("Fetch response status:", res.status);
            if (!res.ok) {
                throw new Error(`HTTP error: ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            console.log("Attendance data:", data);
            lastLoadedCourseId = courseId;

            const list = document.getElementById("attendance-list");
            const title = document.getElementById("attendance-title");
            if (!list || !title) {
                console.error("attendance-list or attendance-title element not found in DOM");
                return;
            }

            title.textContent = ` Lịch sử điểm danh - ${courseName}`;
            list.innerHTML = "";

            if (data.length === 0) {
                list.innerHTML = "<p>Không có dữ liệu điểm danh.</p>";
                return;
            }

            data.forEach(att => {
                const div = document.createElement("div");
                div.className = "list-group-item";
                div.innerHTML = `
                    <strong>${att.student_name}</strong> |
                    <span>${new Date(att.time).toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' })}</span> |
                    ${att.recognized ? "<span class='text-success'> Đã nhận diện</span>" : "<span class='text-danger'> Chưa nhận</span>"}
                    ${att.image_base64 ? `<br><img src="${att.image_base64}" class="rounded mt-2" style="max-height: 100px;">` : ""}
                `;
                list.appendChild(div);
            });
        })
        .catch(err => {
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

function loadCourses() {
    fetch(`/api/lecturer/${lecturerId}/courses`)
        .then(res => res.json())
        .then(data => {
            const courseList = document.getElementById("course-list");
            courseList.innerHTML = "";
            data.forEach(course => {
                const col = document.createElement("div");
                col.className = "col-md-6 mb-3";
                col.innerHTML = `
                    <div class="card shadow-sm">
                        <div class="card-header">
                            <h5 class="card-title mb-0">${course.name}</h5>
                        </div>
                        <div class="card-body">
                            <p><strong>Học kỳ:</strong> ${course.semester}</p>
                            <button class="btn btn-outline-primary btn-sm" onclick="loadStudents(${course.id}, '${course.name}')"> Xem SV</button>
                            <!-- Nút trong phần course card -->
                            <button class="btn btn-outline-secondary btn-sm ms-2"
                                onclick="showTab('attendance', ${course.id}, '${course.name}')">
                                 Lịch sử điểm danh
                            </button>

                        </div>
                    </div>
                `;
                courseList.appendChild(col);
            });
        });
}

function loadStudents(courseId, courseName) {
    currentCourseId = courseId;
    currentCourseName = courseName; // Set the course name
    fetch(`/api/courses/${courseId}/students_with_embedding`)
        .then(res => res.json())
        .then(data => {
            showTab('students');
            document.getElementById("student-title").textContent = ` Danh sách sinh viên - ${courseName}`;
            const studentList = document.getElementById("student-list");
            studentList.innerHTML = "";

            data.forEach(st => {
                const div = document.createElement("div");
                div.className = "list-group-item d-flex justify-content-between align-items-center";

                const info = document.createElement("span");
                info.innerHTML = `<strong>${st.id}</strong> - ${st.full_name}`;

                const btn = document.createElement("button");
                btn.className = `btn btn-sm ${st.has_embedding ? 'btn-warning' : 'btn-primary'}`;
                btn.innerHTML = st.has_embedding ? ' Đăng ký lại' : ' Đăng ký';
                btn.onclick = () => handleRegisterFace(st.id, st.full_name, st.has_embedding);

                div.appendChild(info);
                div.appendChild(btn);

                studentList.appendChild(div);
            });
        });
}
function showQR(courseId, courseName) {
    const qrContainer = document.getElementById("qr-container");
    qrContainer.innerHTML = "";

    const img = document.createElement("img");
    img.src = `/api/qr_ip?course_id=${encodeURIComponent(courseId)}&course_name=${encodeURIComponent(courseName)}`;
    img.alt = "QR chứa thông tin điểm danh";
    img.style.width = "256px";
    img.style.height = "256px";

    img.onload = () => {
        qrContainer.appendChild(img);
        const qrModal = new bootstrap.Modal(document.getElementById('qrModal'));
        qrModal.show();
    };
    img.onerror = () => {
        qrContainer.innerHTML = "<p class='text-danger'>Lỗi: Không thể tải mã QR</p>";
        const qrModal = new bootstrap.Modal(document.getElementById('qrModal'));
        qrModal.show();
    };
}

let currentStudentId = null;
let currentStudentName = null;
let stream = null;

async function handleRegisterFace(studentId, fullName, hasEmbedding) {
    currentStudentId = studentId;
    currentStudentName = fullName;

    const modal = new bootstrap.Modal(document.getElementById('faceModal'));
    modal.show();

    const video = document.getElementById('video');
    try {
        stream = await setupWebcam(video);
    } catch (err) {
        console.error("Failed to setup webcam in handleRegisterFace:", err);
        const status = document.getElementById('register-status');
        status.innerHTML = "<span class='text-danger'> Không thể truy cập webcam</span>";
    }
}

function captureAndSend() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const status = document.getElementById("register-status");
    const ctx = canvas.getContext('2d');

    const capturedImages = [];
    let count = 0;
    const totalCaptures = 10;
    const delay = 600; // 0.6 giây

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    status.innerHTML = "<span class='text-info'> Đang chụp ảnh...</span>";

    const capture = () => {
        ctx.drawImage(video, 0, 0);
        canvas.toBlob(blob => {
            capturedImages.push(blob);
            count++;

            if (count < totalCaptures) {
                setTimeout(capture, delay); // chụp tiếp
            } else {
                // Gửi tất cả ảnh lên server
                const formData = new FormData();
                formData.append("student_id", currentStudentId);
                formData.append("name", currentStudentName);
                for (let i = 0; i < capturedImages.length; i++) {
                    formData.append("files", capturedImages[i], `frame_${i}.jpg`);
                }

                fetch('/register', {
                    method: 'POST',
                    body: formData
                })
                    .then(res => res.json())
                    .then(result => {
                        if (result.success) {
                            status.innerHTML = "<span class='text-success'> Đăng ký thành công!</span>";
                        } else {
                            status.innerHTML = "<span class='text-danger'> Đăng ký thất bại: " + result.message + "</span>";
                        }
                    })
                    .catch(err => {
                        console.error(err);
                        status.innerHTML = "<span class='text-danger'> Lỗi kết nối đến máy chủ</span>";
                    });
            }
        }, 'image/jpeg');
    };

    capture(); // bắt đầu chụp ảnh
}
// Đóng modal và dừng webcam
document.getElementById("faceModal").addEventListener('hidden.bs.modal', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});

async function setupWebcam(videoElement) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        await new Promise(resolve => {
            videoElement.onloadedmetadata = () => {
                videoElement.play(); // Ensure the video plays after metadata is loaded
                resolve();
            };
        });
        return stream;
    } catch (err) {
        alert("Không thể truy cập webcam!");
        console.error("Webcam error:", err);
        throw err; // Rethrow to handle in the calling function
    }
}
async function openAttendanceModal() {
    const video = document.getElementById('attendance-video');
    const canvas = document.getElementById('attendance-canvas');
    const context = canvas.getContext('2d');

    if (!currentCourseId) {
        alert("Vui lòng chọn một khóa học trước khi điểm danh!");
        return;
    }

    if (window.attendanceStream) {
        window.attendanceStream.getTracks().forEach(track => track.stop());
        window.attendanceStream = null;
    }
    if (window.attendanceInterval) {
        clearInterval(window.attendanceInterval);
        window.attendanceInterval = null;
    }

    const modal = new bootstrap.Modal(document.getElementById('attendance-modal'));
    modal.show();

    try {
        const stream = await setupWebcam(video);
        window.attendanceStream = stream;

        video.play();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.clearRect(0, 0, canvas.width, canvas.height);

        let lastRecognition = null;
        let lastRecognitionTime = 0;
        const PERSISTENCE_DURATION = 3000;

        window.attendanceInterval = setInterval(() => {
            const currentTime = Date.now();
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            if (lastRecognition && (currentTime - lastRecognitionTime) < PERSISTENCE_DURATION) {
                const { startX, startY, endX, endY } = lastRecognition.bbox;
                const name = lastRecognition.name;

                const videoWidth = video.videoWidth;
                const videoHeight = video.videoHeight;
                const canvasWidth = canvas.width;
                const canvasHeight = canvas.height;

                const scaleX = canvasWidth / videoWidth;
                const scaleY = canvasHeight / videoHeight;

                const scaledStartX = startX * scaleX;
                const scaledStartY = startY * scaleY;
                const scaledEndX = endX * scaleX;
                const scaledEndY = endY * scaleY;

                context.strokeStyle = 'green';
                context.lineWidth = 2;
                context.beginPath();
                context.rect(
                    scaledStartX,
                    scaledStartY,
                    scaledEndX - scaledStartX,
                    scaledEndY - scaledStartY
                );
                context.stroke();

                context.fillStyle = 'green';
                context.font = '16px Arial';
                context.fillText(name, scaledStartX, scaledStartY - 10);
            }

            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                formData.append('course_id', currentCourseId);

                fetch('/recognize_siamese', {
                    method: 'POST',
                    body: formData
                })
                    .then(res => res.json())
                    .then(data => {
                        if (data.error) {
                            console.warn("Lỗi nhận diện:", data.error);
                            return;
                        }
                        console.log("Recognition result:", data);
                        if (Array.isArray(data) && data.length > 0) {
                            const result = data[0];
                            const name = result.name;
                            showToast(` Đã điểm danh: ${name}`);

                            if (result.bbox) {
                                lastRecognition = result;
                                lastRecognitionTime = Date.now();
                            }
                        }
                    })
                    .catch(err => {
                        console.error("Lỗi gửi API:", err);
                    });
            }, 'image/jpeg', 0.9);
        }, 1000);
    } catch (error) {
        console.error("Không thể mở webcam:", error);
        alert("Không thể mở webcam!");
    }
}
function closeAttendanceModal() {
    const modalEl = document.getElementById('attendance-modal');
    const modal = bootstrap.Modal.getInstance(modalEl);
    modal.hide();

    if (window.attendanceStream) {
        window.attendanceStream.getTracks().forEach(track => track.stop());
        window.attendanceStream = null;
    }

    if (window.attendanceInterval) {
        clearInterval(window.attendanceInterval);
        window.attendanceInterval = null;
    }
    const video = document.getElementById('attendance-video');
    video.srcObject = null;

    // Clear the canvas
    const canvas = document.getElementById('attendance-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

