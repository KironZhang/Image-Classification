````This is a simple but powerful image classification model, hope this will benefit beginners.

In our setup, we will:
1. Data pre-processing:
- Download the data at: 
  https://www.kaggle.com/c/dogs-vs-cats/data
- Create cats-vs-dogs/ folder
- Create training/ and validation/ sub-folders inside cats-vs-dogs/ folder
- Divide the data into cats-vs-dogs/training/ and cats-vs-dogs/validation/ sub-folders
- Create cats/ and dogs/ sub-folders inside cats-vs-dogs/training/ and cats-vs-dogs/validation/, respectively
- Put 11250 cat pictures into training/cats/, 1250 into validation/cats, and put 11250 dog pictures into training/dogs/, 1250 into validation/dogs/
So that we have 11250 training examles for each class, and 1250 validation examples for each class.
In summary, the data structure will be like:
'''
cats-vs-dogs/
    training/
        cats/
            cat000.jpg
            cat001.jpg
            ...
        dogs/
            dog000.jpg
            dog001.jpg
            ...
    validation/
        cats/
            cat000.jpg
            cat001.jpg
            ...
        dogs/
            dog000.jpg
            dog001.jpg
            ...
2. Create simple convolutional neural network for image classification
- training and validation process
3. Test with your own samples
'''
````


        
    
  
