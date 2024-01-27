# Docker

- [1. Common Questions](#1-common-questions)
- [2. Build an image](#2-build-an-image)
- [3. Run an image](#3-run-an-image)
- [4. RUN vs. CMD vs. ENTRYPOINT](#4-run-vs-cmd-vs-entrypoint)
- [5. COPY vs. ADD](#5-copy-vs-add)
- [6. EXPORT vs. SAVE vs. IMPORT](#6-export-vs-save-vs-import)
- [7. Container EXEC](#7-container-exec)
- [8. Docker Repositories](#8-docker-repositories)
- [9. Layers](#9-layers)
  - [9.1. Good Practises](#91-good-practises)
  - [9.2. How to Save Space?](#92-how-to-save-space)
- [10. Housekeeping](#10-housekeeping)
  - [10.1. List containers/images](#101-list-containersimages)
  - [10.2. Stop a container](#102-stop-a-container)
  - [10.3. Remove containers](#103-remove-containers)
  - [10.4. Remove an image](#104-remove-an-image)
- [11. Volumes](#11-volumes)
- [12. Names](#12-names)
- [13. Examples](#13-examples)

<div class="warning">

**Docker image** is a set of instructions we want to create the environment to deploy our application.

**Docker container** is a running instance of the image.

Process with `ID=1` is the most important, e.g., DB or Kernel.

</div>

## 1. Common Questions

What is the difffence between a VM and a container?

> VM has a complete OS and strong isolation, whereas a container provides lightweight isolation and shares the host OS kernel.

Docker vs. Kubernetes?

> Docker is used as the CRE (Container Runtime Environment), while Kubernetes is used as the platform to deploy the images.

## 2. Build an image

In currect working directory:

- `docker [image] build . -t myimagename:1`

With specific project directory:

- `docker [image] build <path_to_dir_with_dockerfile> -t myimagename:1`

With specific Dockerfile and project directory:

- `docker [image] build -f <MyDockerfileName> <path_to_dir_with_dockerfile> -t myimagename:1`

From a GitHub repository:

- `docker [image] build https://github.com/madflojo/automatron.git -t automatron:1`

## 3. Run an image

An image becomes a container when the `docker run` command is executed.

`docker [container] run myimagename:1`

`docker run -d` ...detached mode (in the background)

`docker run -p 8080:80` ...specify ports (expose port 8080 on the local machine / server and 80 on the container)

`docker run -P` ...get a random port on the local machine and expose the port defined in the Dockerfile with `EXPOSE` instruction

## 4. RUN vs. CMD vs. ENTRYPOINT

- `RUN`: executes when **building the image** (e.g., python requirements)

- `CMD` and `ENTRYPOINT`: execute when the **container starts**
  - `ENTRYPOINT`: determines main process to run
  - `CMD`: additional parameters for ENTRYPOINT (can be overriden)

## 5. COPY vs. ADD

These two commands do the same, but...

- `ADD <name>.tar.gz <destination>`: natively supports tarballs (.tar.gz, with "unzipping") and remote URLs; however, it is recommended to use `curl` for remote URLs
- `COPY <src_dir/file> <dest_dir>`
  - e.g. source code or python requirements.txt file

## 6. EXPORT vs. SAVE vs. IMPORT

A simple trick of exporting a  container and importing it as a docker image can save space. See instructions below.

- `EXPORT` - takes a snapshot of the current state of the container
  - `docker export <container_id> > <export_tar_name>.tar`
- `SAVE` - saves the whole history of a container
- `IMPORT`
  - `cat export.tar | docker import - <image_tag>:<version>` ...print raw data of the export and parse it with docker to create a new image

## 7. Container EXEC

How to access a container? Use `exec`! The following command opens a bash terminal:

```bash
docker container exec -it <container_id> /bin/bash
```

- `-i` ...interactive (verbose)
- `-t` ...terminal
- it's often enough to specify the first two characters of `container_id`
- leave the terminal with `exit`

## 8. Docker Repositories

- e.g., [DockerHub](https://hub.docker.com/)

## 9. Layers

- `docker image history <image>`

- `docker image inspect <image>`

Docker containers are building blocks for applications. Each container is an image with a readable/writeable layer on top of a bunch of read-only layers.

These layers (also called intermediate images) are generated when the commands in the Dockerfile are executed during the Docker image build.

TLDR; Layers of a Docker image are essentially just files generated from running some command. You can view the contents of each layer on the Docker host at /var/lib/docker/aufs/diff. Layers are neat because they can be re-used by multiple images saving disk space and reducing time to build images while maintaining their integrity.

- [jessicagreben.medium.com/digging-into-docker-layers](https://jessicagreben.medium.com/digging-into-docker-layers-c22f948ed612)

- each docker command creates a new image layer (immutable!), there is always one writable layer

### 9.1. Good Practises

- Do not use multiple instances of the same instructions (e.g., RUN), each instruction create a new layer and takes up space!

```docker
FROM ubuntu

#Instead of this:
# RUN apt-get update
# RUN apt install curl -y
# RUN apt install ruby -y
# RUN apt install python3 -y
# RUN apt install build-essential -y
# RUN apt install apache2 -y

#Do this (and effectively reduce the number of layers):
RUN apt-get update && \
    apt install curl -y && \
    apt install ruby -y && \
    apt install python3 -y && \
    apt install build-essential -y && \
    apt install apache2 -y
```

### 9.2. How to Save Space?

- Use appropriate base images, e.g. [alpine](https://hub.docker.com/_/alpine/tags) is only about 3 MB, while [ubuntu](https://hub.docker.com/_/ubuntu/tags) is about 25 MB.

- Use `apk  update` and `apk add` instead of `apt-get update` and `apt install`

- Using `apk` and `alpine` can save over 50% space compared to `ubuntu`

```docker
FROM alpine
RUN apk update && \
    apk add curl ruby python3 build-base apache2
```

## 10. Housekeeping

### 10.1. List containers/images

- `docker container ls [-a]`

- `docker container ps`

- `docker image ls`

### 10.2. Stop a container

- `docker container stop <container_id>`

Stop all:

- `docker container stop $(docker container ls -aq)`

### 10.3. Remove containers

Remove all containers:

- `docker container rm $(docker container ls -aq)`

Remove all stoppped containers:

- `docker container prune`

### 10.4. Remove an image

- `docker image rm <id>`

Remove all images (when all containers are stopped):

- `docker image rm $(docker image ls -aq)`

## 11. Volumes

```bash
docker container run -p 8080:80 -v /home/user/mywebsite:/usr/bin/apache2/htdocs httpd:2.4
```

- `-v <from>:<to>`

One can, for example, mount a directory with html files, then these can be updated without building the image again. That is update html file, save it, refresh the webpage and see the changes.

## 12. Names

How to change the tag?

```bash
docker tag myweb:1 mywebserver:1
```

## 13. Examples

Hello World

```docker
FROM alpine:3.14  # or alpine:latest

CMD echo "Hello from Alpine 3.14!"
```

Webserver

```docker
FROM centos:7 # https://hub.docker.com/_/centos

RUN yum -y install httpd

ADD webserver.tar.gz /var/www/html

ENTRYPOINT [ "/usr/sbin/httpd" ]

CMD [ "-D", "FOREGROUND" ]

EXPOSE 80
```
