import cv2
import logging
from flask import Flask, Response, render_template, jsonify, request, send_file
from twilio.rest import Client
from ultralytics import YOLO
import time
import os
import numpy as np
import pandas as pd
import base64
import io
import googlemaps
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger('').addHandler(console)

app = Flask(__name__)


try:
    model_path = "yolov8x.pt"
    if not os.path.exists(model_path):
         logging.error(f"Model file not found at {model_path}")
         model = None
    else:
        model = YOLO(model_path, verbose=False)
        logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    model = None



account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_SID_DEFAULT_PLACEHOLDER")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_TOKEN_DEFAULT_PLACEHOLDER")
RECIPIENT = 'Authorities'

FRAME_AREA = float(os.environ.get("FRAME_AREA", "100.0"))


GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
gmaps = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logging.info("Google Maps client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Google Maps client: {e}")
else:
    logging.warning("GOOGLE_MAPS_API_KEY environment variable not set. Location context features disabled.")



global THRESHOLD, EVENT_TYPE, VIDEO_SOURCE_TYPE, IP_CAMERA_URL, MONITORED_LAT, MONITORED_LON
try:
    THRESHOLD = int(os.environ.get("CROWD_THRESHOLD", "11"))
except ValueError:
    logging.warning("Invalid CROWD_THRESHOLD environment variable. Using default: 2")
    THRESHOLD = 2

EVENT_TYPE = "Public Place"
VIDEO_SOURCE_TYPE = "local"
IP_CAMERA_URL = ""
MONITORED_LAT = None
MONITORED_LON = None


client = None
if account_sid != "YOUR_TWILIO_SID_DEFAULT_PLACEHOLDER" and auth_token != "YOUR_TWILIO_TOKEN_DEFAULT_PLACEHOLDER":
    try:
        client = Client(account_sid, auth_token)
        logging.info("Twilio client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Twilio client: {e}")
else:
    logging.warning("Twilio credentials not found in environment variables. SMS alerts will be disabled.")


last_processed_count = 0
last_alert_time = 0
last_detected_points = []
last_frame_dims = (0, 0)
MAX_HISTORY = 100
history_data = []




@app.route('/')
def landing():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/map_view')
def map_view():
    logging.info("Serving map view page (index2.html)")
    return render_template('index2.html')


@app.route('/heatmap')
def heatmap():
    logging.info("Serving heatmap page (hmap.html)")
    return render_template('hmap.html')


@app.route('/settings')
def settings_page():
    logging.info("Serving settings page (settings.html)")
    return render_template('settings.html')


@app.route('/analytics')
def analytics():
    logging.info("Serving analytics page (analytics.html)")
    return render_template('analytics.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_processed_count, last_alert_time, last_detected_points, last_frame_dims
    global THRESHOLD

    start_time = time.time()

    if not model:
        logging.error("YOLO model not loaded. Cannot process frame.")
        return jsonify({'status': 'error', 'message': 'Model not loaded on server'}), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            logging.warning("Received invalid request data for processing.")
            return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400


        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)


        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
             logging.error("Failed to decode image from received data.")
             return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

        frame_height, frame_width = frame.shape[:2]
        current_detected_points = []


        person_count = 0
        try:
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:

                    if box.cls is not None and len(box.cls) > 0:
                         class_id = int(box.cls[0])
                         if class_id == 0:
                             person_count += 1

                             try:
                                 coords = box.xyxy[0].cpu().numpy().astype(int)
                                 cx = (coords[0] + coords[2]) // 2
                                 cy = (coords[1] + coords[3]) // 2
                                 current_detected_points.append({'x': int(cx), 'y': int(cy), 'value': 1})
                             except Exception as bbox_err:
                                 logging.warning(f"Could not extract bbox/center point: {bbox_err}")
        except Exception as e:
             logging.error(f"Error during YOLO model inference: {e}", exc_info=True)
             return jsonify({'status': 'error', 'message': 'Error during detection'}), 500


        last_processed_count = person_count
        last_detected_points = current_detected_points
        last_frame_dims = (frame_width, frame_height)


        crowd_density = round(person_count / FRAME_AREA, 2) if FRAME_AREA > 0 else 0


        alert_triggered_this_request = False
        current_time = time.time()
        if person_count > THRESHOLD and (current_time - last_alert_time) >= 60:
            alert_triggered_this_request = True
            last_alert_time = current_time
            logging.info(f"Threshold exceeded ({person_count}/{THRESHOLD}). Triggering alert.")


            if client:
                try:

                    client.messages.create(
                    from_='whatsapp:+14155238886',
                    body='🚨Crowd Threshold Exceeded at ExpoMart! Immediate action required!',
                    to='whatsapp:+917368861270'
                    )
                    logging.info(f"SMS alert initiated for {RECIPIENT}.")
                    print("SMS Simulation: Threshold Alert!")
                except Exception as e:
                    logging.error(f"SMS sending failed: {str(e)}")
            else:
                 logging.warning("Threshold exceeded, but Twilio client not available. Cannot send SMS.")

        processing_time = time.time() - start_time
        logging.debug(f"Processed frame in {processing_time:.4f}s. Found {person_count} people.")


        history_entry = {
            'current_count': person_count,
            'max_people': THRESHOLD,
            'density_level': "CRITICAL" if person_count > THRESHOLD else "Normal",
            'alert_active': person_count > THRESHOLD and (current_time - last_alert_time) < 60,
            'last_alert': last_alert_time if last_alert_time else None,
            'timestamp': datetime.now().isoformat(),
            'crowd_density': crowd_density,
            'event_type': EVENT_TYPE,
            'monitored_lat': MONITORED_LAT,
            'monitored_lon': MONITORED_LON
        }
        history_data.append(history_entry)
        if len(history_data) > MAX_HISTORY:
            history_data.pop(0)


        density_level = "CRITICAL" if person_count > THRESHOLD else "Normal"
        alert_active_now = person_count > THRESHOLD and (current_time - last_alert_time) < 60

        return jsonify({
            'status': 'success',
            'person_count': person_count,
            'density_level': density_level,
            'alert_triggered': alert_triggered_this_request,
            'alert_active_cooldown': alert_active_now,
            'threshold': THRESHOLD,
            'processing_time_ms': int(processing_time * 1000),
            'crowd_density': crowd_density,
            'timestamp': datetime.now().isoformat()
        })

    except base64.binascii.Error as b64_error:
        logging.error(f"Base64 decoding error: {b64_error}")
        return jsonify({'status': 'error', 'message': 'Invalid Base64 data'}), 400
    except Exception as e:
        logging.error(f"Error processing frame: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error during frame processing'}), 500


@app.route('/get_heatmap_data')
def get_heatmap_data():
    global last_detected_points, last_frame_dims
    data_to_send = {
        'points': list(last_detected_points),
        'frame_width': last_frame_dims[0],
        'frame_height': last_frame_dims[1],
        'status': 'success' if last_frame_dims[0] > 0 else 'waiting'
    }
    if data_to_send['status'] == 'waiting':
        data_to_send['points'] = []

    return jsonify(data_to_send)


@app.route('/update_settings', methods=['POST'])
def update_settings():

    global THRESHOLD, EVENT_TYPE, VIDEO_SOURCE_TYPE, IP_CAMERA_URL, MONITORED_LAT, MONITORED_LON

    try:
        data = request.get_json()
        if not data:
            logging.warning("Received empty settings update request.")
            return jsonify({'status': 'error', 'message': 'No data received'}), 400


        if 'new_threshold' in data:
            try:
                new_threshold_val = int(data['new_threshold'])
                if new_threshold_val >= 1:
                    THRESHOLD = new_threshold_val
                    logging.info(f"Crowd threshold updated to: {THRESHOLD}")
                else:
                    logging.warning(f"Invalid threshold value received: {data['new_threshold']}. Must be >= 1. Keeping previous value.")
            except (ValueError, TypeError):
                logging.warning(f"Invalid threshold format received: {data['new_threshold']}. Must be an integer. Keeping previous value.")


        if 'event_type' in data:
            EVENT_TYPE = str(data['event_type'])
            logging.info(f"Event type updated to: {EVENT_TYPE}")


        if 'video_source_type' in data:
            source_type = data['video_source_type']
            if source_type in ['local', 'ip']:
                VIDEO_SOURCE_TYPE = source_type
                logging.info(f"Video source type updated to: {VIDEO_SOURCE_TYPE}")

                if source_type == 'ip':
                    raw_url = data.get('ip_camera_url', '')
                    IP_CAMERA_URL = str(raw_url).strip() if raw_url else ""
                    logging.info(f"IP Camera URL updated to: '{IP_CAMERA_URL}'")
                elif source_type == 'local':
                    IP_CAMERA_URL = ""
                    logging.info("Video source set to 'local'. Cleared IP Camera URL.")
            else:
                logging.warning(f"Invalid video_source_type received: {source_type}. Keeping previous value.")


        if 'latitude' in data and 'longitude' in data:
             lat_in = data.get('latitude')
             lon_in = data.get('longitude')
             try:

                 new_lat = float(lat_in) if lat_in is not None and lat_in != '' else None
                 new_lon = float(lon_in) if lon_in is not None and lon_in != '' else None


                 if (isinstance(new_lat, float) and isinstance(new_lon, float)) or \
                    (new_lat is None and new_lon is None):
                     MONITORED_LAT = new_lat
                     MONITORED_LON = new_lon
                     if MONITORED_LAT is not None:
                         logging.info(f"Monitored location updated to Lat: {MONITORED_LAT}, Lon: {MONITORED_LON}")
                     else:
                         logging.info("Monitored location cleared.")
                 else:

                      logging.warning(f"Invalid combination of Lat/Lon received: Lat={lat_in}, Lon={lon_in}. Keeping previous values.")

             except (ValueError, TypeError):

                  logging.warning(f"Invalid latitude/longitude format received: Lat={lat_in}, Lon={lon_in}. Keeping previous values.")

        return jsonify({'status': 'success', 'message': 'Settings updated successfully'})

    except Exception as e:
        logging.error(f"Error updating settings: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error during settings update'}), 500


@app.route('/get_stats')
def get_stats():

    global THRESHOLD, EVENT_TYPE, VIDEO_SOURCE_TYPE, IP_CAMERA_URL, MONITORED_LAT, MONITORED_LON
    global last_processed_count, last_alert_time

    current_time = time.time()

    current_threshold = THRESHOLD if isinstance(THRESHOLD, int) and THRESHOLD >= 1 else 999999
    is_critical = last_processed_count > current_threshold

    alert_active = is_critical and (current_time - last_alert_time) < 60

    return jsonify({
        'current_count': last_processed_count,
        'max_people': THRESHOLD,
        'density_level': "CRITICAL" if is_critical else "Normal",
        'alert_active': alert_active,
        'last_alert': last_alert_time if last_alert_time else None,
        'event_type': EVENT_TYPE,
        'video_source_type': VIDEO_SOURCE_TYPE,
        'ip_camera_url': IP_CAMERA_URL,
        'monitored_lat': MONITORED_LAT,
        'monitored_lon': MONITORED_LON,
        'timestamp': datetime.now().isoformat(),
        'crowd_density': round(last_processed_count / FRAME_AREA, 2) if FRAME_AREA > 0 else 0
    })


@app.route('/get_location_context')
def get_location_context():
    global MONITORED_LAT, MONITORED_LON, gmaps


    if MONITORED_LAT is None or MONITORED_LON is None:
        logging.info("/get_location_context: Monitored location not set.")
        return jsonify({'status': 'no_location', 'message': 'Monitored location not set in settings.'})
    if gmaps is None:
        logging.warning("/get_location_context: Google Maps client not available (API key missing or init failed).")
        return jsonify({'status': 'no_api_key', 'message': 'Google Maps API key not configured on server.'})

    location = (MONITORED_LAT, MONITORED_LON)
    context_data = { 'nearby_pois': [], 'traffic_summary': "N/A", 'status': 'success' }
    errors = []
    logging.info(f"Fetching context for location: {location}")

    try:

        poi_types_keywords = [
            'train_station', 'subway_station', 'bus_station', 'airport', 'transit_station',
            'stadium', 'shopping_mall', 'tourist_attraction', 'movie_theater',
            'bar', 'night_club', 'restaurant', 'cafe'
        ]
        keyword_query = '|'.join(poi_types_keywords)
        search_radius = 500

        logging.debug(f"Performing Nearby Search: location={location}, radius={search_radius}, keyword='{keyword_query}'")
        nearby_places = gmaps.places_nearby(location=location, radius=search_radius, keyword=keyword_query)

        if nearby_places.get('status') == 'OK':
            relevant_pois = []
            all_results = nearby_places.get('results', [])
            logging.debug(f"Nearby Search found {len(all_results)} raw results.")

            for place in all_results:
                place_name = place.get('name')
                place_types = place.get('types', [])
                place_loc = place.get('geometry', {}).get('location')

                if not place_name or not place_loc: continue


                common_types = list(set(place_types) & set(poi_types_keywords))
                if common_types:

                     dist_m = -1
                     try:
                         dist_m = googlemaps.distance.distance(location, (place_loc['lat'], place_loc['lng'])).m
                     except AttributeError:
                         logging.warning("googlemaps.distance module not found (requires version >= 4.4.0). Skipping distance calculation.")
                     except Exception as dist_err:
                         logging.warning(f"Could not calculate distance for {place_name}: {dist_err}")

                     relevant_pois.append({
                        'name': place_name,
                        'types': common_types,
                        'distance': round(dist_m) if dist_m >= 0 else None
                    })


            if relevant_pois and relevant_pois[0].get('distance') is not None:
                relevant_pois.sort(key=lambda x: x['distance'])
            context_data['nearby_pois'] = relevant_pois[:5]
            logging.info(f"Found {len(context_data['nearby_pois'])} relevant nearby POIs.")

        elif nearby_places.get('status') == 'ZERO_RESULTS':
            logging.info("Nearby Search returned ZERO_RESULTS.")
            context_data['nearby_pois'] = []
        else:

             error_msg = nearby_places.get('error_message', 'Unknown Nearby Search error')
             status_code = nearby_places.get('status', 'UNKNOWN_STATUS')
             errors.append(f"Nearby Search Error ({status_code}): {error_msg}")
             logging.error(f"Nearby Search API Error ({status_code}): {error_msg}")



        origin_lat = MONITORED_LAT + 0.018
        origin_lon = MONITORED_LON
        origin = (origin_lat, origin_lon)

        logging.debug(f"Requesting Directions: origin={origin}, destination={location}")
        try:

            directions_result = gmaps.directions(origin, location, mode="driving", departure_time=datetime.now())

            if directions_result and len(directions_result) > 0 and 'legs' in directions_result[0] and len(directions_result[0]['legs']) > 0:
                leg = directions_result[0]['legs'][0]
                duration_sec = leg.get('duration', {}).get('value')
                duration_traffic_sec = leg.get('duration_in_traffic', {}).get('value')


                if duration_sec is not None and duration_traffic_sec is not None and duration_sec > 0:
                    traffic_ratio = duration_traffic_sec / duration_sec
                    logging.debug(f"Traffic calculation: duration={duration_sec}s, duration_traffic={duration_traffic_sec}s, ratio={traffic_ratio:.2f}")
                    if traffic_ratio >= 1.6: context_data['traffic_summary'] = "Heavy"
                    elif traffic_ratio >= 1.25: context_data['traffic_summary'] = "Moderate"
                    else: context_data['traffic_summary'] = "Light/Normal"
                elif duration_sec == 0 :
                     context_data['traffic_summary'] = "N/A (Origin too close)"
                     logging.debug("Traffic calculation skipped: Origin likely too close to destination.")
                else:
                    context_data['traffic_summary'] = "N/A (No traffic data)"
                    logging.debug("Traffic calculation skipped: duration_in_traffic missing.")
            else:
                errors.append("Directions API returned no results or unexpected format.")
                logging.warning("Directions API returned no results or unexpected format.")

        except googlemaps.exceptions.ApiError as e:

             errors.append(f"Directions API Error: {e}")
             logging.error(f"Directions API Error: {e}")
        except Exception as e_dir:

             errors.append(f"Unexpected error during Directions API call: {e_dir}")
             logging.error(f"Unexpected error during Directions API call: {e_dir}", exc_info=True)


    except googlemaps.exceptions.ApiError as e:
        errors.append(f"Google Maps API Error: {e}")
        context_data['status'] = 'api_error'
        logging.error(f"Google Maps API Error: {e}")

    except Exception as e:
        errors.append(f"Unexpected server error fetching location context: {e}")
        context_data['status'] = 'server_error'
        logging.error(f"Error in /get_location_context: {e}", exc_info=True)


    if errors:
        context_data['errors'] = errors

        context_data['status'] = 'error_partial' if context_data['nearby_pois'] or context_data['traffic_summary'] != "N/A" else 'error_full'
        logging.warning(f"Errors encountered in /get_location_context: {errors}")

    return jsonify(context_data)


@app.route('/get_history')
def get_history():
    return jsonify(history_data)


@app.route('/download_data')
def download_data():
    try:
        if not history_data:
            logging.warning("No history data available for export")
            return jsonify({'status': 'error', 'message': 'No data available to download'}), 404

        logging.info(f"Preparing Excel export with {len(history_data)} records")


        export_data = []
        for entry in history_data:
            clean_entry = {
                'timestamp': entry.get('timestamp', ''),
                'current_count': entry.get('current_count', 0),
                'density_level': entry.get('density_level', ''),
                'crowd_density': entry.get('crowd_density', 0),
                'max_people': entry.get('max_people', 0)
            }
            export_data.append(clean_entry)


        df = pd.DataFrame(export_data)


        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logging.error(f"Error formatting timestamps: {e}")


        columns_to_rename = {
            'timestamp': 'Time',
            'current_count': 'People Count',
            'density_level': 'Density Level',
            'crowd_density': 'Crowd Density (people/area)',
            'max_people': 'Maximum Capacity'
        }
        existing_columns = [col for col in columns_to_rename.keys() if col in df.columns]
        rename_map = {col: columns_to_rename[col] for col in existing_columns}
        df = df.rename(columns=rename_map)

        logging.info("Creating Excel file in memory")


        output = io.BytesIO()
        df.to_excel(output, sheet_name='Crowd Data', index=False)
        output.seek(0)


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crowd_data_{timestamp}.xlsx"

        logging.info(f"Sending Excel file as attachment: {filename}")


        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except ImportError as e:
        logging.error(f"Missing required package: {e}")
        return jsonify({'status': 'error', 'message': 'Server missing required packages for Excel export. Please install pandas and openpyxl.'}), 500
    except Exception as e:
        logging.error(f"Error generating Excel file: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to generate Excel file: {str(e)}'}), 500


@app.route('/download_data_csv')
def download_data_csv():
    try:
        if not history_data:
            return jsonify({'status': 'error', 'message': 'No data available to download'}), 404


        export_data = []
        for entry in history_data:
            clean_entry = {
                'Time': entry.get('timestamp', ''),
                'People Count': entry.get('current_count', 0),
                'Density Level': entry.get('density_level', ''),
                'Crowd Density': entry.get('crowd_density', 0),
                'Maximum Capacity': entry.get('max_people', 0)
            }
            export_data.append(clean_entry)


        df = pd.DataFrame(export_data)


        output = io.StringIO()
        df.to_csv(output, index=False)


        bytes_output = io.BytesIO(output.getvalue().encode('utf-8'))


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crowd_data_{timestamp}.csv"


        return send_file(
            bytes_output,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    except Exception as e:
        logging.error(f"Error generating CSV file: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to generate CSV file: {str(e)}'}), 500



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    app.run(host='0.0.0.0', port=port, debug=True)