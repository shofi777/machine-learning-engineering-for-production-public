#!/bin/bash

while sleep 0.01;

do curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://127.0.0.1:49845/v1/models/half_plus_two:predict;

done
