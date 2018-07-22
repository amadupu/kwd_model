import os

if os.name =='nt':
    noise_path = r'D:\codeathon-18\demand'
    alexa_path = r'D:\codeathon-18\alexa'
    alexa_trim_path = r'D:\codeathon-18\alexa_trim'
    mixed_path = r'D:\codeathon-18\mixed'
    valid_train_path = r'D:\codeathon-18\cv_corpus_v1\cv_corpus_v1\cv-valid-train'
    valid_test_path = r'D:\codeathon-18\cv_corpus_v1\cv_corpus_v1\cv-valid-test'
else:
    noise_path = r'/home/arun_madupu/projects/corpus/demand'
    alexa_path = r'/home/arun_madupu/projects/corpus/alexa'
    alexa_trim_path = r'/home/arun_madupu/projects/corpus/alexa/alexa_trim'
    mixed_path = r'/home/arun_madupu/projects/corpus/mixed'
    valid_train_path = r'/home/arun_madupu/projects/corpus/cv_corpus_v1/cv-valid-train'
    valid_test_path = r'/home/arun_madupu/projects/corpus/cv_corpus_v1/cv-valid-test'