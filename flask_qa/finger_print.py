import numpy
from flask import Flask, jsonify, request
import os
import cv2

app = Flask(__name__)




@app.route('/home', methods=['POST'])
def home():
    sample = cv2.imdecode(numpy.frombuffer(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    fingerprint_image = cv2.imdecode(numpy.frombuffer(request.files['file2'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)


    best_score = 0

    image = None

    kp1, kp2, mp = None, None, None

    sift = cv2.SIFT_create()
    keypoints_1, discriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, discriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'tress': 10}, {}).knnMatch(discriptors_1, discriptors_2, k=2)

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100

        image = fingerprint_image

        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    if best_score != None:

        #result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
        #result = cv2.resize(result, None, fx=4, fy=4)
        cv2.waitKey(0)

        result1 = {

            "Bestscore": best_score,
        }
    else:
        result1 = {

            "Bestscore": best_score,

        }

    cv2.destroyAllWindows()
    return jsonify(result1)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
