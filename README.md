# EG_BERT_NER
   
A named entity recognition for English corpus build on the basis of google-research/bert and run on tf1.13-Gpu version.        
    
The deploy code for generating saved pb for tensorflow serving and online prediction are also included.     


## DEPLOYMENT
       
(1) Use export_saved_pb.py to generate savedmodel pd for tensorflow serving.   
    
(2) Copy savedmodel pb into tensorflow serving docker and copy tf_serving_entrypoint.sh into it also,the details     
can be seen on goole tensorflow official net.        
   
(3) Finaly, we can use bert_ber_client.py to predict result. Tensorflow serving stands for grpc and http request.   


