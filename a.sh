#!/bin/bash
stringContain() { case $2 in *$1* ) return 0;; *) return 1;; esac ;}

echo $1
if stringContain "driving" $1; then
	echo "match"
fi

echo "done"
