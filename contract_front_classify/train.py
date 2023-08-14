import glob
import os
import pickle
import random
import tensorflow_models as tfm
import tensorflow as tf
import cv2
import numpy as np
from seqeval.metrics import accuracy_score
from sklearn import svm

from sklearn.model_selection import train_test_split

from contract_front_classify.parameter import Parameter


def get_model_x(input_shape1, input_shape2, output_shape=3):
    in_tensor_txt = tf.keras.layers.Input(shape=input_shape1)
    x = tf.keras.layers.Flatten()(in_tensor_txt)
    x = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    txt_feature = tf.keras.layers.Dense(64, activation='sigmoid')(x)

    in_tensor_img = tf.keras.layers.Input(shape=input_shape2)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(in_tensor_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    img_feature = tf.keras.layers.Dense(64, activation='sigmoid')(x)

    x = tf.keras.layers.concatenate([txt_feature, img_feature])
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=[in_tensor_txt, in_tensor_img], outputs=x)
    return model


def get_model_img(input_shape, output_shape):
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    mobilenet = tfm.vision.backbones.MobileNet(
        model_id='MobileNetV3Small',
        filter_size_scale=1.0,
        input_specs=tf.keras.layers.InputSpec(shape=input_shape),
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        output_stride=None,
        min_depth=8,
        divisible_by=8,
        stochastic_depth_drop_rate=0.0,
        regularize_depthwise=False,
        use_sync_bn=False,
        finegrain_classification_mode=True,
        output_intermediate_endpoints=False,
    )
    x = mobilenet(inputs)
    x = x['6']
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def get_model_txt(input_shape1, output_shape):
    in_tensor_txt = tf.keras.layers.Input(shape=input_shape1)
    x = tf.keras.layers.Flatten()(in_tensor_txt)
    x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=in_tensor_txt, outputs=x)
    return model


def class1_acc(y_true, y_pred):
    if y_true.shape[1] == 2:
        mask = tf.constant([1, 1], dtype=tf.float32)
    else:
        mask = tf.constant([1, 1, 0], dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)
    y_pred = tf.multiply(y_pred, mask)
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


if __name__ == '__main__':

    acc_tf = []
    acc_svm = []
    param = Parameter()
    if param.use_text:
        text_input = np.load('./text_input.npy')
        if param.use_img:
            img_input = np.load('./img_input.npy')
        else:
            img_input = np.zeros((text_input.shape[0], 1, 1))
    else:
        if param.use_img:
            img_input = np.load('./img_input.npy')
            text_input = np.zeros((img_input.shape[0], 1, 1))
        else:
            print('At least one inputs must be valid')
            exit()
    label = np.load('./label.npy')
    if param.output_shape == 2:
        label[label == 2] = 0
    img_name = pickle.load(open('./img_name.pkl', 'rb'))
    for _ in range(20):
        seed = random.randint(0, 1000)
        x1_train, x1_test, x2_train, x2_test, y_train_raw, y_test_raw, img_name_train, img_name_test = train_test_split(
            text_input, img_input, label, img_name,
            random_state=seed, train_size=0.7)
        if param.tensorflow_backend:
            if len(acc_tf) > 0:
                continue
            # tensorflow mix model
            # import tensorflow as tf

            y_train = tf.one_hot(y_train_raw, param.output_shape)
            y_test = tf.one_hot(y_test_raw, param.output_shape)
            if param.use_img and param.use_text:
                model = get_model_x(input_shape1=(x1_train.shape[1], 1),
                                    input_shape2=(param.img_size, param.img_size, 3),
                                    output_shape=param.output_shape)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=class1_acc)
                model.summary()
                model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=32,
                          validation_data=([x1_test, x2_test], y_test),
                          callbacks=[
                              tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5', save_best_only=True)])
            elif param.use_text and not param.use_img:

                model = get_model_txt(input_shape1=(x1_train.shape[1], 1)
                                      , output_shape=param.output_shape)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=class1_acc)

                model.summary()
                history = model.fit(x1_train[:, :, np.newaxis], y_train, epochs=100, batch_size=32,
                                    validation_data=(x1_test[:, :, np.newaxis], y_test),
                                    callbacks=[
                                        tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',
                                                                           save_best_only=True)])
            elif param.use_img and not param.use_text:
                model = get_model_img(input_shape=(1, param.img_size, param.img_size, 3), output_shape=param.output_shape)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=class1_acc)
                model.summary()
                history = model.fit(x2_train, y_train, epochs=100, batch_size=32,
                                    validation_data=(x2_test, y_test),
                                    callbacks=[
                                        tf.keras.callbacks.ModelCheckpoint('./contract_models/model.h5',
                                                                           save_best_only=True)])

            acc_tf.append(max(history.history['val_class1_acc']))

        # 'sigmoid', 'poly', 'precomputed', 'rbf', 'linear'
        # 'ovr', 'ovo'
        if param.svm_backend and not param.use_img:
            clf = svm.SVC(C=0.85, kernel='linear', decision_function_shape='ovr', verbose=False)
            clf.fit(x1_train, y_train_raw)
            y_pred = clf.predict(x1_test)
            y_pred[y_pred == 2] = 0
            y_test_raw[y_test_raw == 2] = 0
            y_pred_train = clf.predict(x1_train)
            y_pred_train[y_pred_train == 2] = 0
            y_train_raw[y_train_raw == 2] = 0
            if param.show_res:
                for i in np.where(y_test_raw != y_pred)[0]:
                    print(f"'{img_name_test[i]}'")
                    img_path = glob.glob(os.path.join(param.root, '*', img_name_test[i]))[0]
                    img = cv2.imread(img_path)
                    label = img_path.split('/')[-2]
                cv2.imshow(f'{label}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                print('-------------------')
                for i in np.where(y_train_raw != y_pred_train)[0]:
                    print(img_name_train[i])
                img_path = glob.glob(os.path.join(param.root, '*', img_name_train[i]))[0]
                img = cv2.imread(img_path)
                label = img_path.split('/')[-2]
                cv2.imshow(f'{label}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print('all test num : ', len(y_test_raw))

                print(accuracy_score(y_test_raw, y_pred))
                print(accuracy_score(y_train_raw, y_pred_train))
            acc_svm.append(accuracy_score(y_test_raw, y_pred))
            with open('./model_loc.pkl', 'wb') as f:
                pickle.dump(clf, f)
    if len(acc_tf) > 0:
        print(np.mean(acc_tf))
    if len(acc_svm) > 0:
        print(np.mean(acc_svm))
