import tensorflowjs as tfjs
import tensorflow as tf
from keras.models import load_model
tf.compat.v1.disable_eager_execution()

model_age = load_model('./models/age.h5')
tfjs.converters.save_keras_model(model_age, './models/model_age')
model_sex = load_model('./models/sex.h5')
tfjs.converters.save_keras_model(model_age, './models/model_sex')
model_name = load_model('./models/age.h5')
tfjs.converters.save_keras_model(model_age, './models/model_name')
model_breed = load_model('./models/breed.h5')
tfjs.converters.save_keras_model(model_breed, './models/model_breed')
model_context = load_model('./models/age.h5')
tfjs.converters.save_keras_model(model_context, './models/model_context')