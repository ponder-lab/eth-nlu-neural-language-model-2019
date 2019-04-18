docker pull brunnelu/tensorflow:latest
docker run -it --rm -p 80:8888 -v $PWD:/tf -w /tf brunnelu/tensorflow:latest
