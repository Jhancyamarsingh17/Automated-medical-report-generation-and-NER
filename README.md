# Automatic X-ray Report Generation using Deep Learning Models and Named Entity Recognition

Deep learning has found an important application area in healthcare due to its quick and effective development in research domains.
This project combines the power of computer vision(CV) and natural language processing(NLP). 

The project objective is to automatically generate reports that is as close to one generated by a radiologist as possible provided one or more medical X-Ray Images of a patient  and  identifying the medical entities using Natural language Processing and Deep Learning techniques. 
We have used different state-of-art techniques for this task like CNN (InceptionV3, CheXNet ) and RNN(LSTM,GRU)   with the Attention Mechanism as the Attention model mainly focuses on the important image features/words thus improving the prediction of text report. Lastly, we have attempted to Identify the medical entities like disease,organs, biochemicals from the textual descriptions of the generated X-ray report using Named Entity Recognition technique. We have used pretrained SpaCy model trained on biomedical data. Specifically, Spcispacy is Python package containing spaCy models for processing biomedical, scientific or clinical text. 


DATASET : The dataset used is publicly available on Indian University (IU) datasets.The original source of this data is Open-i service: National Library of Medicine (https://openi.nlm.nih.gov/faq). This dataset contain 7471 images, which are images of chest X - Ray (frontal, lateral), report of the corresponding image in XML file.

The repository has following folders:
1) Data - The data folder has all the necessary pretrained model weights file , dependencies , curated data.
2) EDA - This folder has file showing Exploratory data Analysis on IU dataset. 
3) Model_1_GRU_with_Inception - This folder has the model which is trained on one image using Encoder-Inception v3 pre-trained model and Decoder-GRU with Attention   mechanism. The generated report has been tagged and simplified using NER model.
4) Model_2_GRU_with_CheXNet - This folder has the model which is trained on one image using Encoder-CheXNet pre-trained model and Decoder-GRU with Attention mechanism. Also, using the NER model the resulting report was labeled and simplified.
5) Model_3_LSTM_with_CheXNet -  This folder has the model which is trained on concatenated features of two image using Encoder-CheXNet pre-trained model and Decoder-LSTM with Attention mechanism implemented along with NER.
7) Deployment - This folder has the streamlit file to showcase the web interface of the working model.

To run the model, navigate to Deployment folder , clone it and use the following command : 

- streamlit run medical_report_generator.py

![streamlit_SC](https://user-images.githubusercontent.com/93669922/206935080-0e4e33be-83d8-4952-82a9-9bec5cce5b56.png)











