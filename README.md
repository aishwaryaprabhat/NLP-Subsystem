# NLP-Subsystem
1.  'requirements.txt' contains the frozen requirements 
2. 'Integrator_with_sass.py' contains the speech subsystem code integrated into the API functions
3. 'test_sass.py' contains  some lines of code to test. It loads the 'genevieve.wav' audio file and returns the predictions and last fully connected layers of the 3 models in the subsystem. To run, run the command 'python test_sass.py'
4. The .pickle files are the tokenizer objects needed to tokenize new text according to the format of the trained models
5. The .h5 files are the neural network models 
