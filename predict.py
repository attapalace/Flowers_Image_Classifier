from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(
    description='This is a prediction program to predict flowers names for flowers images')
parser.add_argument('--image',default  ='./test_images/cautleya_spicata.jpg',
                    help='The image of flower to be predicted')
parser.add_argument('--model',default = './my_trained_model.h5',
                    help='A pretrained model to predict the image')
parser.add_argument('--top_k',type=int,default=1,
                    help='Return the top K most likely classes')
parser.add_argument('--category_names',default='label_map.json',
                    help='Path to a JSON file mapping labels to flower names')


args = vars(parser.parse_args())

category_names = args['category_names']
k = args['top_k']
image_path = args['image']
saved_model = args['model']

model = tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(image):
    tf_image = tf.cast(image, dtype=tf.float32)
    new_image = tf.image.resize(tf_image,[224,224])
    new_image /=255
    return new_image.numpy()

def predict(image_path,model,k):
    im = Image.open(image_path)
    np_image = np.asarray(im)
    image = process_image(np_image)
    expanded_image = np.expand_dims(image,axis=0)
    
    ps = model.predict(expanded_image)
    probs, classes = tf.math.top_k(ps, k=k)
    return probs.numpy(), classes.numpy()

#loading in a mapping from label to category name from label_map.json, to produce the actual names of the flowers.
with open(category_names, 'r') as f:
    class_names = json.load(f)
    
probs , classes = predict(image_path,model,k)
flower_names=[]
for i in classes[0] :
    flower_names.append(class_names[str(i+1)])

print(f'The predicted flower classes are {flower_names} with probabilities {probs}')

#useful resources 
#https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/
