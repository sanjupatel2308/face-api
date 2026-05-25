import base64
import requests
import os

BASE_URL = "http://127.0.0.1:5000"


def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def print_result(title, res):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print("Status Code:", res.status_code)
    try:
        print("Response:", res.json())
    except Exception:
        print("Raw Response:", res.text)


def test_health():
    res = requests.get(f"{BASE_URL}/health")
    print_result("HEALTH TEST", res)


def test_detect(image_path, title):
    image_b64 = file_to_base64(image_path)
    res = requests.post(
        f"{BASE_URL}/detect",
        json={
            "imageBase64": image_b64
        }
    )
    print_result(title, res)


def test_register():
    image_b64 = file_to_base64("test_images/1.jpeg")
    res = requests.post(
        f"{BASE_URL}/face/register",
        json={
            "userId": "test123",
            "name": "Test User",
            "imageBase64": image_b64
        }
    )
    print_result("REGISTER TEST", res)


def test_verify_same_person():
    image_b64 = file_to_base64("test_images/2.jpeg")
    res = requests.post(
        f"{BASE_URL}/face/verify",
        json={
            "userId": "test123",
            "imageBase64": image_b64
        }
    )
    print_result("VERIFY SAME PERSON TEST", res)


def test_verify_wrong_person():
    image_b64 = file_to_base64("test_images/3.jpeg")
    res = requests.post(
        f"{BASE_URL}/face/verify",
        json={
            "userId": "test123",
            "imageBase64": image_b64
        }
    )
    print_result("VERIFY WRONG PERSON TEST", res)


if __name__ == "__main__":
    if not os.path.exists("test_images/1.jpeg"):
        print("❌ test_images/1.jpeg not found")
    elif not os.path.exists("test_images/2.jpeg"):
        print("❌ test_images/2.jpeg not found")
    elif not os.path.exists("test_images/3.jpeg"):
        print("❌ test_images/3.jpeg not found")
    else:
        test_health()

        test_detect("test_images/1.jpeg", "DETECT REGISTER IMAGE")
        test_detect("test_images/2.jpeg", "DETECT VERIFY SAME IMAGE")
        test_detect("test_images/3.jpeg", "DETECT VERIFY WRONG IMAGE")

        test_register()
        test_verify_same_person()
        test_verify_wrong_person()