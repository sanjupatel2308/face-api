import base64
import requests
import os
import time

BASE = "http://127.0.0.1:5000"
IMG_DIR = "test_images"

def b64(f):
    p = os.path.join(IMG_DIR, f)
    with open(p, "rb") as x:
        return base64.b64encode(x.read()).decode()

def test():
    # Health check
    print("1. Health:", requests.get(f"{BASE}/health").json())
    
    # Register test user
    print("\n2. Register:", requests.post(f"{BASE}/face/register", json={
        "userId": "test999", "name": "Test", "imageBase64": b64("1.jpeg")
    }).json())
    
    # === OLD ENDPOINT (2 frames) ===
    print("\n3. OLD endpoint (2 frames):")
    r = requests.post(f"{BASE}/face/liveness-verify", json={
        "userId": "test999",
        "frameClosedBase64": b64("1.jpeg"),  # same photo = no real blink
        "frameOpenBase64": b64("2.jpeg")
    }).json()
    print(f"   live={r.get('live')}, matched={r.get('matched')}, msg={r.get('msg')}")
    
    # === NEW STRONG ENDPOINT (5 frames) ===
    print("\n4. NEW STRONG endpoint (5 frames):")
    # Same photo 5 times = static spoof
    r = requests.post(f"{BASE}/face/liveness-verify-strong", json={
        "userId": "test999",
        "framesBase64": [b64("1.jpeg"), b64("1.jpeg"), b64("1.jpeg"), b64("1.jpeg"), b64("1.jpeg")]
    }).json()
    print(f"   live={r.get('live')}, matched={r.get('matched')}")
    print(f"   msg={r.get('msg')}")
    print(f"   layer={r.get('layer')}")
    print(f"   checks={r.get('checks')}")
    
    # Cleanup
    requests.post(f"{BASE}/face/delete", json={"userId": "test999"})
    print("\n✅ Test complete")

if __name__ == '__main__':
    test()