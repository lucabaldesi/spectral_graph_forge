#!/bin/bash
NAME=$1

if [ -z $NAME ]; then
	echo "Pass a network name"
else
	mkdir $NAME
	for ((i=0; i<10; i++)) do
		./benchmark
		mv network.dat "$NAME/$NAME-$i.edges"
	done
fi
