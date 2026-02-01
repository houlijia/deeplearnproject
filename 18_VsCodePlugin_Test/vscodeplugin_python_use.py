import requests
import json

API_BASE_URL = "http://localhost:3000"


def trigger_uppercase():
    """向 VSCode 插件 API 发送请求以触发大写转换功能"""
    url = f"{API_BASE_URL}/HelloWorld"

    # 可以发送一些数据，如果插件需要的话
    payload = {
        # "some_data": "example_value"
    }

    try:
        response = requests.post(url)  # 使用 POST 方法
        response.raise_for_status()  # 检查 HTTP 错误

        result = response.json()
        print("Response from VSCode extension:", result)

        if result.get('success'):
            print("Command executed successfully!")
        else:
            print(f"Command failed: {result.get('error')}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling VSCode API: {e}")


def check_status():
    """检查 API 服务器状态"""
    url = f"{API_BASE_URL}/status"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        print("Status check response:", result)
    except requests.exceptions.RequestException as e:
        print(f"Error checking status: {e}")


if __name__ == "__main__":
    print("Checking status...")
    check_status()

    print("\nHelloWorld command...")
    trigger_uppercase()

