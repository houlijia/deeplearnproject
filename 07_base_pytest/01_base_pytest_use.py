# test_pytest_features.py
import pytest
import sys

# 基础测试示例
def test_addition():
    assert 1 + 1 == 2

# 1. 异常断言
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# 2. 参数化测试
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_addition_parametrized(a, b, expected):
    assert a + b == expected

# 3. Fixture 使用
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

# 4. 共享 Fixture (conftest.py 中定义)
def test_shared_fixture(shared_resource):
    assert shared_resource.startswith("Shared")

# 5. 标记和筛选
@pytest.mark.slow
def test_long_running():
    import time
    time.sleep(2)
    assert True

# 6. 跳过测试
@pytest.mark.skip(reason="功能已弃用")
def test_old_feature():
    assert False

# 7. 条件跳过
@pytest.mark.skipif(sys.version_info < (3, 8), reason="需要 Python 3.8+")
def test_python38_feature():
    assert hasattr(sys, 'get_coroutine_origin_tracking_depth')

# 8. 猴子补丁
def test_monkeypatch(monkeypatch):
    monkeypatch.setenv("TEST_ENV", "pytest")
    assert "pytest" in os.getenv("TEST_ENV")

# 9. 临时目录
def test_tmpdir(tmpdir):
    temp_file = tmpdir.join("test.txt")
    temp_file.write("pytest")
    assert temp_file.read() == "pytest"

# 10. 捕获输出
def test_capsys(capsys):
    print("Hello pytest")
    captured = capsys.readouterr()
    assert "pytest" in captured.out

# 11. 测试类
class TestMathOperations:
    def test_multiplication(self):
        assert 3 * 4 == 12

    def test_division(self):
        assert 8 / 2 == 4

# 12. Fixture 作用域
@pytest.fixture(scope="module")
def db_connection():
    print("\n建立数据库连接")
    yield "DB:Connected"
    print("\n关闭数据库连接")

def test_db_query(db_connection):
    assert "Connected" in db_connection

# 13. 预期失败
@pytest.mark.xfail
def test_beta_feature():
    assert False, "这是实验性功能"

# 14. 插件使用 (pytest-mock)
def test_mocking(mocker):
    mock_obj = mocker.patch('os.getcwd')
    mock_obj.return_value = "/mock/dir"
    assert os.getcwd() == "/mock/dir"

# 15. 自定义标记
@pytest.mark.integration
def test_api_call():
    # 实际中这里会有真实的 API 调用
    assert True

# 16. Fixture 参数化
@pytest.fixture(params=[1, 2, 3])
def number(request):
    return request.param

def test_number_square(number):
    assert number ** 2 >= number