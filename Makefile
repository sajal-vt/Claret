GCC=g++
F:=1
N:=0
S:=0
INCLUDE=/usr/local/cuda-7.5/include

all: bin/glimmer

bin/glimmer: bin/glimmer.o
	$(GCC) -I ./bin -I $(INCLUDE) -lOpenCL bin/glimmer.o -o $@ -lm

bin/glimmer.o: source/host/cpp/glimmer_host.cpp
	$(GCC) -I  ./bin -I $(INCLUDE) -c source/host/cpp/glimmer_host.cpp -o bin/glimmer.o

run: bin/glimmer
	cd ./bin ; rm -vf *.out log *.csv; ./glimmer $(F) $(N) $(S) | tee log; cd -

clean:
	rm -f bin/*.o bin/*.mod bin/glimmer

.PHONY:
	run clean
