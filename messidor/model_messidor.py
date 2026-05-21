import tensorflow as tf

def create_model(input_shape=(224,224,3), classes=5, lr=1e-4):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg'
    )
    base.trainable = False

    x = tf.keras.layers.Dense(256, activation='relu')(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(base.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model