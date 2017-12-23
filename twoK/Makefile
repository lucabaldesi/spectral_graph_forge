CC=g++
CCFLAGS=-O3 -march=native -w -Wno-deprecated -D USE_UNORDERED

CPPFILES=2k_simple.cpp

all:	test

test:	$(CPPFILES)
	$(CC)  $(CCFLAGS) $(CPPFILES) -o 2k_simple

clean:
	rm -f 2k_simple

.PHONY: all clean

