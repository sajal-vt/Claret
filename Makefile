GCC=g++
F:=1
N:=0
S:=0
INCLUDE=/usr/local/cuda-9.0/include

all: bin/claret

bin/claret: bin/claret.o
	$(GCC) -I ./bin -I $(INCLUDE) -lOpenCL bin/claret.o -o $@ -lm

bin/claret.o: source/host/cpp/claret.cpp
	$(GCC) -I  ./bin -I $(INCLUDE) -c source/host/cpp/claret.cpp -o bin/claret.o

clean:
	rm -f bin/*.o bin/*.mod bin/claret

.PHONY:
	run clean
