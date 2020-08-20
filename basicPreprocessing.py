from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

test = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
