import base64
import requests
import os
import time

BASE = "http://127.0.0.1:5000"
IMG_DIR = "test_images"


def b64(filename):
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def show(label, res):
    ok = res.get('success') or res.get('live') or False
    icon = "✅" if ok else "❌"
    print(f"  {icon} {label}: {res.get('msg', res.get('message', 'N/A'))}")
    if 'confidence' in res:
        print(f"     Confidence: {res['confidence']}%")
    if 'eyeState' in res:
        print(f"     Eye State: {res['eyeState']} ({res.get('detail','')})")
    if 'live' in res:
        print(f"     Liveness: {'PASSED' if res['live'] else 'FAILED'}")
    if 'matched' in res:
        print(f"     Face Match: {'YES' if res['matched'] else 'NO'}")
    return res


def wait_server():
    print("Server wait kar raha hu...", end=" ")
    for i in range(10):
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                print("OK!")
                return True
        except:
            pass
        time.sleep(1)
    print("FAILED!")
    return False


def main():
    if not wait_server():
        return

    results = []

    # TEST 1: Health
    print("\n" + "="*50 + "\nTEST 1: Health\n" + "="*50)
    r = requests.get(f"{BASE}/health").json()
    show("Health", r)
    results.append(('Health', r.get('success')))

    # TEST 2: Register
    print("\n" + "="*50 + "\nTEST 2: Register (1.jpeg)\n" + "="*50)
    img = b64('1.jpeg')
    if img:
        r = requests.post(f"{BASE}/face/register", json={
            'userId': 'test_user_1', 'name': 'Test User', 'imageBase64': img
        }).json()
        show("Register", r)
        results.append(('Register', r.get('success')))

    # TEST 3: Verify same
    print("\n" + "="*50 + "\nTEST 3: Verify SAME (2.jpeg)\n" + "="*50)
    img = b64('2.jpeg')
    if img:
        r = requests.post(f"{BASE}/face/verify", json={
            'userId': 'test_user_1', 'imageBase64': img
        }).json()
        show("Verify Same", r)
        results.append(('Verify Same', r.get('matched')))

    # TEST 4: Verify wrong
    print("\n" + "="*50 + "\nTEST 4: Verify WRONG (3.jpeg)\n" + "="*50)
    img = b64('3.jpeg')
    if img:
        r = requests.post(f"{BASE}/face/verify", json={
            'userId': 'test_user_1', 'imageBase64': img
        }).json()
        show("Verify Wrong", r)
        results.append(('Verify Wrong', not r.get('matched')))

    # TEST 5: Check eyes
    print("\n" + "="*50 + "\nTEST 5: Check Eyes (1.jpeg)\n" + "="*50)
    img = b64('1.jpeg')
    if img:
        r = requests.post(f"{BASE}/face/check-eyes", json={
            'imageBase64': img
        }).json()
        show("Eyes Check", r)
        results.append(('Eyes Check', r.get('eyeState') in ['open', 'closed']))

    # TEST 6: Liveness correct
    print("\n" + "="*50 + "\nTEST 6: Liveness + Match\n" + "="*50)
    open_img = b64('1.jpeg')
    closed_img = b64('4.jpeg')
    if not closed_img:
        print("  ⚠️ 4.jpeg nahi mila! Same person ki eyes-closed photo daalo.")
    else:
        r = requests.post(f"{BASE}/face/liveness-verify", json={
            'userId': 'test_user_1',
            'frameClosedBase64': closed_img,
            'frameOpenBase64': open_img,
        }).json()
        show("Liveness+Match", r)
        results.append(('Liveness+Match', r.get('live') and r.get('matched')))

    # TEST 7: Liveness no blink
    print("\n" + "="*50 + "\nTEST 7: No Blink Rejected\n" + "="*50)
    i1, i2 = b64('1.jpeg'), b64('2.jpeg')
    if i1 and i2:
        r = requests.post(f"{BASE}/face/liveness-verify", json={
            'userId': 'test_user_1', 'frameClosedBase64': i1, 'frameOpenBase64': i2,
        }).json()
        show("No Blink", r)
        results.append(('No Blink', not r.get('live')))

    # TEST 8: Delete
    print("\n" + "="*50 + "\nTEST 8: Delete\n" + "="*50)
    r = requests.post(f"{BASE}/face/delete", json={'userId': 'test_user_1'}).json()
    show("Delete", r)
    results.append(('Delete', r.get('success')))

    # REPORT
    print("\n" + "="*50 + "\nFINAL REPORT\n" + "="*50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  TOTAL: {passed}/{total}")
    if passed == total:
        print("  🎉 ALL PASSED!")
    else:
        print("  ⚠️ Kuch fail hue")
    print()


if __name__ == '__main__':
    main()