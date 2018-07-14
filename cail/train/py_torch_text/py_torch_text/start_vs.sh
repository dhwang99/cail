curdir=`pwd`
nohup python -m visdom.server -port 5900 -env_path ${curdir}/html/  &
