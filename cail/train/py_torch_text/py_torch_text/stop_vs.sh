`ps aux|grep visdom.server | grep -v grep|awk '{print "kill "$2}'`
