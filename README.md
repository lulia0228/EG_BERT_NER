# EG_BERT_NER
   
A named entity recognition for English corpus build on the basis of google-research/bert.     

## DEPLOYMENT
The deploy code for generate saved pb for tensorflow serving is also included.       
    
(1) Use export_saved_pb.py to genarate saved model pd for tensorflow serving.   
    
(2) Copy pb into tensorflow serving docker and copy tf_serving_entrypoint.sh into it also,the details     
can be seen on goole tensorflow official net.        
   
(3) Finaly, we can use bert_ber_client.py to predict result. Tensorflow serving stands for grpc and http request.   


