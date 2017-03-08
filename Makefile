grimson: grimson.cpp
	g++ -O3 `pkg-config --cflags opencv` -o `basename grimson.cpp .cpp` grimson.cpp `pkg-config --libs opencv` -std=c++11

clean:
	rm -f grimson foreground/* background/*

.PHONY: clean
