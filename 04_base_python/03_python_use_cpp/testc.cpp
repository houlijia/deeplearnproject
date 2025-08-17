// example.cpp
extern "C" {
    int add(int a, int b){
        return a+b;
    }
}


// g++ -shared -fPIC -o libmath_operations.so math_operations.cpp   --linux
// g++ -shared -o math_operations.dll math_operations.cpp  --window
