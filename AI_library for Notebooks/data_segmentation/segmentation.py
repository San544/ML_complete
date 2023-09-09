import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
IMG_WIDTH=128
IMG_HIEGHT=128
IMG_CHANNELS=3

inputs=tf.keras.layers.Input((IMG_WIDTH,IMG_HIEGHT,IMG_CHANNELS))
s=tf.keras.layers.Lambda(lambda x:x/255)(inputs)

#contraction path
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(s)
c1=tf.keras.layers.Dropout(0.1)(c1)
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c1)
p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p1)
c2=tf.keras.layers.Dropout(0.1)(c2)
c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c2)
p2=tf.keras.layers.MaxPooling2D((2,2))(c2)

c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p2)
c3=tf.keras.layers.Dropout(0.2)(c3)
c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c3)
p3=tf.keras.layers.MaxPooling2D((2,2))(c3)

c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p3)
c4=tf.keras.layers.Dropout(0.2)(c4)
c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c4)
p4=tf.keras.layers.MaxPooling2D((2,2))(c4)


c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(p4)
c5=tf.keras.layers.Dropout(0.3)(c5)
c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c5)
p5=tf.keras.layers.MaxPooling2D((2,2))(c5)


#expansive path
u6=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
u6=tf.keras.layers.concatenate([u6,c4])
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u6)
c6=tf.keras.layers.Dropout(0.2)(c6)
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c6)

u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7=tf.keras.layers.concatenate([u7,c3])
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u7)
c7=tf.keras.layers.Dropout(0.2)(c7)
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c7)

u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8=tf.keras.layers.concatenate([u8,c2])
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u8)
c8=tf.keras.layers.Dropout(0.1)(c8)
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c8)

u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9=tf.keras.layers.concatenate([u9,c1])
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(u9)
c9=tf.keras.layers.Dropout(0.1)(c9)
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(c9)

outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)

model=tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()


##################################

checkpointer=tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5',verbose=1,save_best_only=True)

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')

]

model.fit(X_train,y_train,batch_size=16,epochs=25,validation_split=0.2,callbacks=callbacks)
