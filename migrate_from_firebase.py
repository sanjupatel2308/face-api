import os
import json
import shutil
import requests
from datetime import datetime
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, firestore

FACES_DIR = 'registered_faces'
TMP_DIR = 'temp_files'

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def init_firebase():
    cred_path = 'serviceAccountKey.json'
    if not os.path.exists(cred_path):
        print("\n" + "❌"*30)
        print("ERROR: serviceAccountKey.json not found!")
        print("\nKaise laaye:")
        print("1. Firebase Console jao")
        print("2. Project Settings → Service Accounts")
        print("3. 'Generate new private key' dabao")
        print("4. Download hui JSON file ko yaha 'serviceAccountKey.json' naam se save karo")
        print("❌"*30 + "\n")
        exit(1)

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


def download_and_register(user_id, name, photo_url):
    tmp_path = None
    try:
        if not photo_url or not photo_url.startswith('http'):
            print(f"  ⚠️  {user_id}: Invalid photoURL")
            return False

        print(f"  ⏳ {user_id}: Downloading...")

        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(photo_url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"  ❌ {user_id}: Download failed (HTTP {r.status_code})")
            return False

        # Temp save
        tmp_path = os.path.join(TMP_DIR, f'mig_{user_id}.jpg')
        with open(tmp_path, 'wb') as f:
            f.write(r.content)

        # Face check
        faces = DeepFace.extract_faces(
            img_path=tmp_path,
            detector_backend='opencv',
            enforce_detection=False
        )
        valid = [f for f in faces if float(f.get('confidence', 0)) > 0.5]

        if len(valid) == 0:
            print(f"  ❌ {user_id}: No face detected")
            return False
        if len(valid) > 1:
            print(f"  ❌ {user_id}: Multiple faces detected")
            return False

        # Save to registered_faces
        udir = os.path.join(FACES_DIR, user_id)
        if os.path.exists(udir):
            shutil.rmtree(udir)

        os.makedirs(udir, exist_ok=True)
        face_path = os.path.join(udir, 'face.jpg')
        shutil.copy2(tmp_path, face_path)

        # Meta save
        with open(os.path.join(udir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'userId': user_id,
                'userName': name,
                'at': datetime.now().isoformat(),
                'source': 'cloudinary'
            }, f, indent=2)

        print(f"  ✅ {user_id}: Registered ({name})")
        return True

    except Exception as e:
        print(f"  ❌ {user_id}: Error - {str(e)}")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("\n" + "="*60)
    print("  MIGRATE: Firebase/Cloudinary → Face Server")
    print("="*60 + "\n")

    db = init_firebase()

    # Firestore se users fetch karo
    print("📥 Firestore se users fetch kar raha hu...")
    users_ref = db.collection('users')
    docs = users_ref.stream()

    users = []
    for doc in docs:
        data = doc.to_dict()
        role = data.get('role', '')

        # Admin ko skip karo (unhe attendance nahi lagani)
        if role == 'admin':
            continue

        photo_url = data.get('photoURL', '')
        if not photo_url:
            continue

        users.append({
            'userId': doc.id,
            'name': data.get('name', 'Unknown'),
            'photoURL': photo_url,
            'role': role
        })

    print(f"Found {len(users)} users (employees + managers) with photoURL\n")

    success = 0
    failed = 0

    for user in users:
        uid = user['userId']
        name = user['name']
        url = user['photoURL']

        if download_and_register(uid, name, url):
            success += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print(f"  DONE: {success} success, {failed} failed")
    print(f"  Total registered faces: {len(os.listdir(FACES_DIR))}")
    print("="*60 + "\n")

    # List all registered
    print("Registered users:")
    for uid in os.listdir(FACES_DIR):
        meta_path = os.path.join(FACES_DIR, uid, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                m = json.load(f)
                print(f"  • {uid}: {m.get('userName', 'N/A')}")


if __name__ == '__main__':
    main()