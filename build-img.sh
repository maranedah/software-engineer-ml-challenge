#!/bin/bash

project_id=latam-challenge-398100 
project_name=software-engineer-ml-challenge
base_tag=gcr.io/$project_id/$project_name
img_name=deploy
version=latest

docker build -t $base_tag/$img_name:$version .
docker push $base_tag/$img_name:$version