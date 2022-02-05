#!/bin/bash -x
activate () {
    . /home/ubuntu/BERT-Backend-Flask/venv/bin/activate
}

runbert () {
    python3 /home/ubuntu/BERT-Backend-Flask/main.py
}

activate
runbert
echo gaaaaaaaaaaaaaaaaa

