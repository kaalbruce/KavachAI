import cv2
import logging
from flask import Flask, Response, render_template, jsonify
from twilio.rest import Client
from ultralytics import YOLO
from tqdm import tqdm
import winsound
import threading
import time

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger('').addHandler(console)

app = Flask(__name__)
model = YOLO("yolov8n.pt", verbose=False).to("cuda")

# Configuration
account_sid = "YOUR_TWILIO_SID"
auth_token = "YOUR_TWILIO_TOKEN"
client = Client(account_sid, auth_token)
PHONE_NUMBER = "+1234567890"
RECIPIENT = "+0987654321"
THRESHOLD = 5  # Changed to match HTML's capacity limit

# Global state variables
current_count = 0
sms_sent = False  # Track SMS alert status

def detect_people():
    global current_count, sms_sent
    cap = cv2.VideoCapture(0)
    pbar = tqdm(desc="Processing", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, device="cuda", verbose=False)
            person_count = 0

            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:
                        person_count += 1

            current_count = person_count  # Update global count

            # SMS Alert Logic
            if person_count > THRESHOLD and not sms_sent:
                try:
                    # client.messages.create(
                    #     body=f"🚨 ALERT! Crowd threshold exceeded! Current count: {person_count}",
                    #     from_=PHONE_NUMBER,
                    #     to=RECIPIENT
                    # )
                    print("hello world")
                    sms_sent = True  # Prevent duplicate alerts
                    logging.info("SMS alert sent successfully")
                except Exception as e:
                    logging.error(f"SMS failed: {str(e)}")

            # Reset SMS flag when crowd reduces
            if person_count <= THRESHOLD:
                sms_sent = False

            # Visual overlays
            if person_count > THRESHOLD:
                frame = cv2.putText(
                    frame, "WARNING: CROWD OVERLOAD", 
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )
                # Trigger siren
                threading.Thread(
                    target=lambda: winsound.Beep(3000, 850) * 3,
                    daemon=True
                ).start()

            frame = cv2.putText(
                frame, f"People: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            pbar.update(1)

    except Exception as e:
        logging.error(f"Detection failed: {str(e)}")
    finally:
        cap.release()
        pbar.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        detect_people(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_stats')
def get_stats():
    density_level = "CRITICAL" if current_count > THRESHOLD else "Normal"
    return jsonify({
        'current_count': current_count,
        'max_people': THRESHOLD,
        'density_level': density_level,
        'alert_active': current_count > THRESHOLD,
        'last_alert': time.time() if current_count > THRESHOLD else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)