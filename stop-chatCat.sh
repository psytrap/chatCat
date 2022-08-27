#!/bin/sh

ps -o pid,args | grep chatCat | grep -v grep | awk '{print "kill -9 " $1}' 

