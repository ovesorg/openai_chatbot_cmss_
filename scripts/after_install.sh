#!/bin/bash

sudo chown -R ubuntu:ubuntu /home/ubuntu
cd /home/ubuntu/openai_chatbot_cmss_
# sudo docker-compose stop
# sudo docker-compose rm -f
# node dist/main
# sudo docker stop auth-microservice
# sudo docker rm auth-microservice
# sudo docker rmi $(docker images -q)
sudo docker pull ghcr.io/ovesorg/chatbot:dev
export DOCKER_CLIENT_TIMEOUT=300
export COMPOSE_HTTP_TIMEOUT=300    
sudo docker-compose -f docker-compose.yml up -d --build
