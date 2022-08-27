#!/bin/sh

ps -o pid,args | grep "python3 -u ./chatCat.py" | grep -v grep | awk '{print "kill -9 " $1}' 

