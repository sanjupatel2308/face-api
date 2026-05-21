from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import json
import shutil
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Folders
FACES_DIR = 'registered_faces'
os.makedirs(FACES_DIR, exist_ok=True)

def save_base64_image(base64_string, filepath):
    """Save base64 image to file"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    with open(filepath, 'wb') as f:
        f.write(image_data)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'service': 'Face Recognition API (DeepFace)',
        'version': '2.0.0',
    })

@app.route('/detect', methods=['POST'])
def detect_face():
    """Check if face exists in photo"""
    try:
        data = request.json
        image_base64 = data.get('image', '')

        if not image_base64:
            return jsonify({'success': False, 'message': 'No image'})

        # Save temp image
        temp_path = 'temp_detect.jpg'
        save_base64_image(image_base64, temp_path)

        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend='opencv',
                enforce_detection=False
            )

            face_count = len([f for f in faces if f['confidence'] > 0.5])

            if face_count == 0:
                return jsonify({
                    'success': False,
                    'faceDetected': False,
                    'faceCount': 0,
                    'message': 'No face detected'
                })

            if face_count > 1:
                return jsonify({
                    'success': False,
                    'faceDetected': True,
                    'faceCount': face_count,
                    'message': f'{face_count} faces. Sirf ek chahiye.'
                })

            face = faces[0]
            return jsonify({
                'success': True,
                'faceDetected': True,
                'faceCount': 1,
                'confidence': round(face['confidence'] * 100, 2),
                'faceLocation': face['facial_area'],
                'message': 'Face detected ✅'
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/register', methods=['POST'])
def register_face():
    """Register employee face"""
    try:
        data = request.json
        user_id = data.get('userId', '')
        user_name = data.get('userName', '')
        image_base64 = data.get('image', '')

        if not user_id or not image_base64:
            return jsonify({'success': False, 'message': 'userId and image required'})

        # Save face image
        user_dir = os.path.join(FACES_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        face_path = os.path.join(user_dir, 'face.jpg')
        save_base64_image(image_base64, face_path)

        # Verify face exists in image
        try:
            faces = DeepFace.extract_faces(
                img_path=face_path,
                detector_backend='opencv',
                enforce_detection=True
            )
            if len(faces) == 0:
                os.remove(face_path)
                return jsonify({'success': False, 'message': 'No face in photo'})
        except Exception as e:
            if os.path.exists(face_path):
                os.remove(face_path)
            return jsonify({'success': False, 'message': f'Face detection failed: {str(e)}'})

        # Save metadata
        meta_path = os.path.join(user_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'userId': user_id,
                'userName': user_name,
                'registeredAt': datetime.now().isoformat(),
            }, f)

        return jsonify({
            'success': True,
            'message': f'{user_name} ka face register ho gaya ✅',
            'userId': user_id,
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/verify', methods=['POST'])
def verify_face():
    """Verify face for attendance"""
    try:
        data = request.json
        user_id = data.get('userId', '')
        image_base64 = data.get('image', '')

        if not user_id or not image_base64:
            return jsonify({'success': False, 'matched': False, 'message': 'userId and image required'})

        # Check registered face exists
        registered_path = os.path.join(FACES_DIR, user_id, 'face.jpg')
        if not os.path.exists(registered_path):
            return jsonify({
                'success': False,
                'matched': False,
                'message': 'Face not registered'
            })

        # Save live photo temp
        temp_path = f'temp_verify_{user_id}.jpg'
        save_base64_image(image_base64, temp_path)

        try:
            # Compare faces
            result = DeepFace.verify(
                img1_path=registered_path,
                img2_path=temp_path,
                model_name='VGG-Face',
                detector_backend='opencv',
                enforce_detection=False,
            )

            matched = result['verified']
            distance = result['distance']
            threshold = result['threshold']
            confidence = round(max(0, (1 - distance / threshold)) * 100, 2)

            # Load user name
            meta_path = os.path.join(FACES_DIR, user_id, 'meta.json')
            user_name = ''
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    user_name = meta.get('userName', '')

            return jsonify({
                'success': True,
                'matched': matched,
                'confidence': confidence,
                'distance': round(distance, 4),
                'threshold': round(threshold, 4),
                'userName': user_name,
                'message': f'Face matched ✅ ({confidence}%)' if matched else f'Face NOT matched ❌ ({confidence}%)',
                'timestamp': datetime.now().isoformat(),
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'success': False, 'matched': False, 'message': f'Error: {str(e)}'})

@app.route('/delete', methods=['POST'])
def delete_face():
    """Delete registered face"""
    try:
        data = request.json
        user_id = data.get('userId', '')
        user_dir = os.path.join(FACES_DIR, user_id)

        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            return jsonify({'success': True, 'message': 'Face deleted'})

        return jsonify({'success': False, 'message': 'User not found'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/list', methods=['GET'])
def list_users():
    """List registered users"""
    users = []
    if os.path.exists(FACES_DIR):
        for uid in os.listdir(FACES_DIR):
            meta_path = os.path.join(FACES_DIR, uid, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    users.append(meta)

    return jsonify({'success': True, 'users': users, 'count': len(users)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Face API running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)