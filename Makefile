build: Dockerfile
	docker build -t mlhw5 .

run: build
	docker run -d --rm --name mlhw5c -p 8888:8888 -v `pwd`:/home/jovyan/work mlhw5 start-notebook.sh --NotebookApp.token=''

stop:
	docker stop mlhw5c

terminal:
	docker run --rm -it -v `pwd`:/home/jovyan/work mlhw5 /bin/bash

clean:
	docker rmi mlhw5