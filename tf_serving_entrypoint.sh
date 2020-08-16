#!/bin/bash 

tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=bert_ner --model_base_path=/models/bert_ner
