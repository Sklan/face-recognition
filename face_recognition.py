import cv2
import numpy as np
import tensorflow as tf

from detect import find_face


def load_graph(file_name):
    with open(file_name, 'rb') as f:
        content = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def predict(result_tensor):
    prediction = tf.argmax(result_tensor, 1)
    return prediction


labels = dict()
with open('data/output_labels.txt', 'r') as f:
    temp = f.readlines()
    for i in range(len(temp)):
        labels.update({i: temp[i]})

graph = load_graph('data/output_graph.pb')
image_buffer_input = graph.get_tensor_by_name('Mul:0')
final_tensor = graph.get_tensor_by_name('final_result:0')
with tf.Session(graph=graph) as sess:
    video_camera = cv2.VideoCapture(0)
    while video_camera.isOpened():
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        ret, frame = video_camera.read()
        cropped, bb = find_face(frame, 299)
        if bb:
            cropped = np.reshape(cropped, newshape=(1, 299, 299, 3))
            prediction = predict(final_tensor)
            predicted_label = sess.run(prediction, feed_dict={image_buffer_input: cropped})
            print labels[predicted_label[0]]
            x1, x2, y1, y2 = None, None, None, None
            for box in bb:
                x1 = box.left()
                y1 = box.top()
                x2 = box.right()
                y2 = box.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, labels[predicted_label[0]],
                            org=(int(x2) - 50, int(y2) + 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=0.2, thickness=1,
                            fontScale=0.4)
        cv2.imshow('Camera', frame)
